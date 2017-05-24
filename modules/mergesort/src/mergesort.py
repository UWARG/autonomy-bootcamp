
def merge_sort(a): # 
	n = len(a)
	if n == 1:
		return a

	halfLength = len(a)//2
	a1 = [a[i] for i in range(0,halfLength)]
	a2 = [a[i] for i in range(halfLength,n)]

	a1 = merge_sort(a1)
	a2 = merge_sort(a2)


	return merge(a1,a2)


def merge(a,b):
	c = []

# this block takes the first element of the sorted list 
# and adds it to the new list.
	while (len(a)!=0 and len(b)!=0): 
		if (a[0] < b[0]):
			c.append(a[0])
			del a[0]
		else:
			c.append(b[0])
			del b[0]

	while (len(a)!=0):
		c.append(a[0])
		del a[0]

	while (len(b)!=0):
		c.append(b[0])
		del b[0]


	return c

def main():
	a = [14,7,8,10,1,15,2]
	assert(merge_sort(a) == [1, 2, 7, 8, 10, 14, 15])
	print a
	print merge_sort(a)

if __name__=="__main__":
	main()

