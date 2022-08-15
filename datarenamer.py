import os

# assign directory
directory = 'data/masks'
i = 0


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        i += 1
        os.rename(f, directory + '/'+ str(i) + ".jpg")