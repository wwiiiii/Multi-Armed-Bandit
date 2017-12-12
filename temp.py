from multiprocessing import Pool

def f(x, y):
	return x+y

def main():
	with Pool(5) as p:
		print(p.map(f, [*(1, 2), *(3, 4)]))

if __name__ == '__main__':
	main()
