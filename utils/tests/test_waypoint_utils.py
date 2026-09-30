"""
TODO(bootcamper): write the tests for ``src/waypoint_utils.py`` in here.

The example below covers files that parse fine: with and without ``home``,
and files with comments and blank lines in them. The rest is yours:

- Bad data: a file whose top level isn't a mapping, waypoints missing
  ``lat``, ``lon``, or ``alt``, values that aren't numbers, YAML that
  doesn't parse, and a file that isn't there.
- Out of range: latitudes past +/-90 and longitudes past +/-180 get
  rejected.
- Nothing to work with: an empty file, an empty ``waypoints`` list, and
  ``sort_clockwise_sweep`` given a list of 0 or 1 waypoints.
- ``east_north_coordinate_offset_m``: offsets you worked out yourself,
  compared with ``pytest.approx``. Never use ``==`` on meters.
- Ordering: with no ``home``, ``sort_clockwise_sweep`` goes clockwise
  starting from north.
- With a ``home``: the order starts in home's direction instead, and goes
  back to starting at north if home is right on top of the centroid.
- Two waypoints in the same direction: the closer one comes first.
- Parsing gives you frozen ``Coordinate`` objects that can't be changed.

Graded by ``warg run utils grade-tests``: pass on the real code, 90% branch
coverage, and fail on every broken copy in ``grader/mutants/``.
"""

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)

# The helper and the test below are given to you.


def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and hand back its path.

    ``tmp_path`` is a pytest fixture: a fresh empty directory per test.
    """
    #tmp_path là 1 thư mục tạm trống cho mỗi test
    #text: nội dung muốn ghi vào file
    path = tmp_path / "waypoints.yaml"
    #tạo ra 1 đường dẫn đến file rỗng đó
    path.write_text(text)
    #ghi nội dung text vào file đó
    return path
  #trả về đường dẫn file dùng tạm để cho trsst cú pháp parse_waypoints_file
#===> tạo nội dung muốn test => nó sẽ tạo 1 file , cho cú pháp b muốn test sau đó là tạo file để test


# One test, three files. ``parametrize`` runs the test body once per
# ``(text, expected)`` pair, and ``ids`` names each run so a failure tells you
# which file broke.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: 8, alt: 9}
            """,
            (None, [Coordinate(4, 5, 6), Coordinate(7, 8, 9)]),
        ),
        (
            """
            # a lap

            home: {lat: 1, lon: 2, alt: 3}

            waypoints:
              # first leg
              - {lat: 4, lon: 5, alt: 6}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
    ],
    ids=["home-and-waypoints", "no-home", "comments-and-blank-lines"],
)
#tạo trước 3 cặp giá trị text - expect để sau đó để xuống dưới đi test
def test_parse_waypoints_file_success(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == expected
    #tạo 1 file trống cho text, ném vào cú pháp parse_waypoints_file => nếu giống với expected thì là qua

# BAD DATA & PARSING ERRORS ------------------------------------------------------
#a file that isn't there
def test_parse_waypoints_file_missing(tmp_path):
    path = tmp_path / "tmp_path.yaml"
    with pytest.raises(OSError):
        parse_waypoints_file(path)

#YAML that doesn't parse
def test_parse_waypoints_file_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: 1, lon: [invalid") # chưa được parse chuẩn
    with pytest.raises(ValueError, match = "invalid YAML"):
        parse_waypoints_file(path)

# a file whose top level isn't a mapping
def test_parse_waypoints_file_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "123")
    with pytest.raises(ValueError, match="expected a mapping"):
      parse_waypoints_file(path)

#a file đúng cú pháp home với waypoints nhưng bên trong home nó là str chứ kph dict
def test_parse_waypoints_file_home_not_mapping(tmp_path):
    text = """
    home: "not a dict"
    waypoints: []
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_waypoints_file(path)

#waypoints missing ``lat``, ``lon``, or ``alt``
def test_parse_waypoints_file_missing_key(tmp_path):
  text = """
  waypoints:
  - {lat: 4, lon: 5}
  """
  path = write_to_tmp_waypoints_file(tmp_path, text)
  with pytest.raises(ValueError, match = "missing key"):
    parse_waypoints_file(path)  

# values that aren't numbers
def test_parse_waypoints_file_non_numeric_values(tmp_path):
  text = """
  waypoints:
  - {lat: 'abc', lon: 5, alt: '6'}
  """
  path = write_to_tmp_waypoints_file(tmp_path, text)
  with pytest.raises(ValueError, match="non-numeric value"):
    parse_waypoints_file(path)

#Không có home (Vẫn hợp lệ) nhưng waypoints là str thay vì lists
def test_parse_waypoints_file_waypoints_not_a_list(tmp_path):
    text = """
    waypoints: "not a list"
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="'waypoints' must be a list"):
        parse_waypoints_file(path)

#OUT OF RANGES ----------------------------------------
"""Out of range: latitudes past +/-90 and longitudes past +/-180 get
  rejected."""
@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (91.0, 0.0),
        (-91.0, 0.0),
        (0.0, 181.0),
        (0.0, -181.0),
    ],
)

def test_parse_waypoints_file_out_of_range(tmp_path, lat, lon):
  text = f"""
    waypoints:
    - {{lat: {lat}, lon: {lon}, alt: 10}}
    """
  path = write_to_tmp_waypoints_file(tmp_path, text)
  with pytest.raises(ValueError, match="latitude/longitude out of range"):
    parse_waypoints_file(path)

#có thể làm 4 trường hợp với 4 hàm khác nhau => parametrize tối ưu hơi

#NOTHING TO WORK WITH -----------------------------------------------------------
"""- Nothing to work with:  
  ``sort_clockwise_sweep`` given a list of 0 or 1 waypoints."""

#an empty file => hàm có trả về đúng (NONE) => dùng assert
def test_parse_waypoints_file_waypoints_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])
#HÀM LỖI => with pytest ...
#HÀM ĐÚNG => assert
#an empty ``waypoints`` list
def test_parse_waypoints_file_waypoints_empty_file_one(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """
    home: {lat: 1, lon: 2, alt: 3}
    waypoints: []
    """)
    assert parse_waypoints_file(path) == (Coordinate(1,2,3), [])
#cú pháp: (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)])

#sort_clockwise_sweep`` given a list of 0 or 1 waypoints.
#tại nó đang yêu cầu sắp xếp sao cho con này bay effectively => chỉ có 1 điểm thì làm j có j mà sort
#case o có coordinate
def test_coordinate_empty():
   result = sort_clockwise_sweep([])
   assert result == []

#case có 1 coordinate

def test_coordinate_one():
   result = sort_clockwise_sweep([Coordinate(1, 2, 3)])
   assert result == [Coordinate(1, 2, 3)]

#OFFSET CALCULATION -------------------------------------------------
"""``east_north_coordinate_offset_m``: offsets you worked out yourself,
  compared with ``pytest.approx``. Never use ``==`` on meters."""
def test_east_north_coordinate_offset_m_same_point():
   east, north = east_north_coordinate_offset_m(10.0, 20.0, 10.0, 20.0)
   assert east == pytest.approx(0.0, abs=1e-6)
   assert north == pytest.approx(0.0, abs=1e-6)

def test_east_north_coordinate_offset_m_north_only():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 1.0, 0.0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(111195.1, abs=0.1)
#test trường hợp north

def test_east_north_coordinate_offset_m_east_at_nonzero_lat():
    east, north = east_north_coordinate_offset_m(60.0, 0.0, 60.0, 1.0)
    assert north == pytest.approx(0.0, abs=1e-6)
    assert east == pytest.approx(55597.5, abs=1.0)
#test cả 2 (lat lon)

#test nếu quên đổi từ độ sang rad
def test_east_north_coordinate_offset_m_north_non_zero_lat():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 1.0, 0.0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(111195.1, rel=1e-2)
    assert north < 200000.0
#Quên nhân cos (họ nghĩ mỗi độ là khoảng cách như nhau, tuy nhiên mỗi độ ở gần cực lại gần hơn là xa cực => phải nhân cos)
#bot chạy qua vì lần trc để cos 0 = 1 => như nhau nếu k có => phải để 60


#ORDERING -------------------------------------------------------------------
"""Ordering: with no ``home``, ``sort_clockwise_sweep`` goes clockwise
  starting from north."""
def test_sort_clockwise_sweep_no_home():
   north = Coordinate(1, 0, 0)
   east = Coordinate(0, 1, 0)
   south = Coordinate(-1, 0, 0)
   west = Coordinate(0, -1, 0)

   result = sort_clockwise_sweep([east, north, west, south])
   assert result == [north, east, south, west]
#home không tồn tại

"""With a ``home``: the order starts in home's direction instead, and goes
  back to starting at north if home is right on top of the centroid."""
def test_sort_clockwise_sweep_with_home():
   north = Coordinate(1, 0, 0)
   east = Coordinate(0, 1, 0)
   south = Coordinate(-1, 0, 0)
   west = Coordinate(0, -1, 0)
   home = Coordinate(0, 2, 0)
   result = sort_clockwise_sweep([east, north, west, south], home=home)
   assert result == [east, south, west, north]
#home có tồn tại + k trùng centroid

def test_sort_clockwise_sweep_with_home_at_centroid():
   north = Coordinate(1, 0, 0)
   east = Coordinate(0, 1, 0)
   south = Coordinate(-1, 0, 0)
   west = Coordinate(0, -1, 0)

   home = Coordinate(0, 0, 0)
   result = sort_clockwise_sweep([east, north, west, south], home=home)
   assert result == [north, east, south, west]
#có home và nó ở centroid => đi từ default từ north => east => south => west

"""Two waypoints in the same direction: the closer one comes first."""
#2 điểm cùng hướng SO VỚI TÂM CỦA CHÚNG => cái nào gần hơn thì đi trc
def test_sort_clockwise_sweep_with_same_direction():
   near_north = Coordinate(2, 0, 0)
   far_north = Coordinate(4, 0, 0)
   south = Coordinate(-6, 0, 0)

   home = Coordinate(0, 0, 0)
   result = sort_clockwise_sweep([south, far_north, near_north], home=home)
   assert result == [near_north, far_north, south]


"""Parsing gives you frozen ``Coordinate`` objects that can't be changed."""
#case home nằm đúng centroid + frozen
def test_parse_waypoints_file_returns_frozen_coordinates(tmp_path):
  text = """
    home: {lat: 1, lon: 2, alt: 3}
    waypoints:
      - {lat: 4, lon: 5, alt: 6}
    """
  path = write_to_tmp_waypoints_file(tmp_path, text)
  home, waypoints = parse_waypoints_file(path)

  with pytest.raises(AttributeError):
     home.lat = 69.0
  with pytest.raises(AttributeError):
     waypoints[0].alt = 70.0