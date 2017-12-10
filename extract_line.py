
import random
import os
import argparse

def get_all_files(path):
	return [os.path.join(path, filename) for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename)) and not filename.startswith('.')]

def extract_line(args):
	files = get_all_files(args.input)

	for filename in files:
		try:
			with open(filename, 'r') as fp:
				lines = fp.readlines()
				if len(lines) <= args.num_limit:
					lines_new = lines
				else:
					lines_new = random.sample(lines, args.num_limit)
				parent_name, child_name = os.path.split(filename)
				filename_o = os.path.join(args.output_dir, child_name)
				file_o = open(filename_o, 'w')
				for line in lines_new:
					file_o.write(line + '\r\n' + '\r\n')
				file_o.close()
				print 'Successfully selected and write file:', child_name
		except Exception, e:
			raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required = True, help = 'dir of the clustering files')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output clustering files')
    parser.add_argument('-n', '--num_limit', type=int, required = True, help = 'The maximum number of lines in a file')
    args = parser.parse_args()
    extract_line(args)

if __name__ == '__main__':
    main()