# WARG Autonomy Bootcamp

This bootcamp teaches you the tools we use to write the software that flies our drones: the `warg` cli, sparse checkouts, project manifests, pytest, behavior trees, and a full drone mission flown against a simulator.

Everything you build goes into **one pull request** that you keep adding to across 5 parts, and that a lead reviews at the end.

## The five parts

| Part | What you do                                    | Where                          |
| ---- | ---------------------------------------------- | ------------------------------ |
| 1    | Bootstrap the repo, create the `intro` project | this README                    |
| 2    | Write `SimCamera`, based on a example          | [`camera/`](camera/)           |
| 3    | Write tests for our waypoint utilities         | [`utils/`](utils/)             |
| 4    | Build a perception behavior tree               | [`airside/`](airside/)         |
| 5    | Fly the mission on SITL                        | [`integration/`](integration/) |

Every part comes with a finished, working example sitting right next to the thing you have to build. Read it first.

## Prerequisites

- **git** and a GitHub account
- **[uv](https://docs.astral.sh/uv/)** - manages Python and dependencies
- **warg-cli** - `uv tool install warg-cli`
- **Docker** - only needed for Part 5 (and optional container runs in Part 4)
- **GitHub CLI (`gh`)** - optional, makes forking easier

Run `warg --help` (or `warg <command> --help`) at any point to see everything the cli can do.

## Part 1: Bootstrap

### 1. Fork and clone

If you have GitHub CLI (`gh`) installed and authenticated, run:

```bash
warg bootcamp
```

and follow its instructions. Otherwise (or if that command isn't available in your warg-cli version yet), do it manually:

1. Fork this repository on GitHub: click **Fork** on `UWARG/autonomy-bootcamp`.
2. Clone **your fork** with warg (this uses a sparse checkout, more on that later):

   ```bash
   warg clone git@github.com:<your-username>/autonomy-bootcamp.git
   cd autonomy-bootcamp
   git remote add upstream git@github.com:UWARG/autonomy-bootcamp.git
   ```

3. Check your environment:

   ```bash
   warg doctor
   ```

### 2. Create your branch

```bash
git status
git checkout -b bootcamp
```

All your work goes on this `bootcamp` branch.

### 3. Meet the registry

```bash
warg list
```

Every project in this repo is listed in [`projects.toml`](projects.toml) at the root. An entry is just a name and a folder:

```toml
[projects.intro]
path = "intro"
```

Each project then describes itself in its own `warg.toml` file. Here's the whole file for the project you're about to make, copy it exactly:

```toml
name = "intro"
description = "Bootcamp introduction project."

depends_on = []
```

(The other projects also have a `[commands]` section. You'll start using those in Part 2.)

### 4. Create the `intro` project

1. Make the directory `intro/` with two files:
   - `intro/warg.toml` - the manifest shown above.
   - `intro/README.md` - containing:
     - your full name,
     - your Waterloo email,
     - your GitHub username.

2. Register it: add the `[projects.intro]` entry shown above to the root `projects.toml`.

3. Verify:

   ```bash
   warg list          # should now include intro
   warg info intro    # should print your manifest
   ```

### 5. Commit, push, open your PR

```bash
git status
git add projects.toml intro
git commit -m "Add intro project"
git push -u origin bootcamp
```

Then open a **draft pull request** on GitHub:

```text
<your-username>:bootcamp  →  UWARG/autonomy-bootcamp:main
```

CI runs on your PR and the **intro** check should go green. This PR is your submission, and every part after this one adds more commits to it.

## Working through Parts 2–5

Check out each project when you get to its part:

```bash
warg up camera        # Part 2
warg up utils         # Part 3
warg up airside       # Part 4
warg up integration   # Part 5
```

Run project commands with:

```bash
warg run <project> <command>     # e.g. warg run camera test
```

After each part, commit and push to update your PR:

```bash
git add <changed files>
git commit -m "<describe the change>"
git push
```

Open a project's README when you start its part. It has the full instructions.

### `warg up` and sparse checkouts

You might have noticed that `warg list` shows projects that aren't anywhere in your folder. That's on purpose. `warg clone` didn't download the whole repo, it did a **sparse checkout**: git got the history, but only wrote some of the files to disk. The other projects are registered, they're just not on your computer yet.

`warg up <project>` adds a project to the list of things git keeps on disk, so its folder shows up. It also brings in whatever that project needs, which is the `depends_on` list in its `warg.toml`, and whatever those need, and so on. `warg up integration` gets you `airside` and `sitl`, plus `camera` and `utils` because `airside` needs them.

`warg down <project>` takes a project back off your disk. Nothing is lost when you do this, it's all still in git, and `warg up` brings the files right back.

Why do it this way: the real WARG repo is much bigger than this one. Only checking out what you're working on keeps clones fast and stops your editor and your search results from filling up with code you don't care about.

```bash
warg up <project>       # put a project (and what it needs) on disk
warg down <project>     # take it back off
warg list               # every project and what it depends on
warg info <project>     # one project's description, dependencies, and commands
```

Run `warg --help` or `warg <command> --help` to see the rest of the cli.

## Getting help

Ask a lead in your bootcamp thread! The bootcamp assumes you know some Python but may be brand new to git, pytest, behavior trees, ROS, and SITL. Getting stuck is normal, staying stuck is optional.
