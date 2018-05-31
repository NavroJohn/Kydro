from osgeo import gdal
import numpy as np
import glob
import os


def read_images(path, imgformat):

    """
    Function to read images having provided extension.
    :param path: Directory to read.
    :param imgformat: Image format.
    :return: Dictionary of images where the key is an image year and the value is a GDAL Image Object.
    """

    print("Reading images...")
    images = {}
    files = os.path.join(path, imgformat)  # create file pattern
    for file in glob.glob(files):  # for each file matching the above pattern
        pos = file.rfind(os.sep) + 1  # get the last file separator position
        year = int(file[pos : pos + 4])  # extract year from file name
        images[year] = gdal.Open(file)  # add GDAL object for the specific year
    print("Finished reading")
    return images


def read_input(years):

    """
    Reads user input
    :param years: Years for which image files are available
    :return: Tuple containing all the validated inputs
    """

    '''
    Sample input is as follows...
    Enter minimum distance: 50
    Enter a valid year: 2010
    Enter number of rows/cols in transition matrix: 5
    Enter distance interval: 2
    Enter 0/1 for AP/GP: 1
    '''
    print("\nEnter the following inputs...")
    min_dist = int(input('Enter minimum distance: '))
    if min_dist < 0:
        raise ValueError('Invalid input')
    year = int(input('Enter a valid year: '))
    if year not in years:
        raise ValueError("Invalid year")
    terms = int(input('Enter number of rows/cols in transition matrix: '))
    dist_interval = int(input('Enter distance interval: '))
    if terms < 1 or dist_interval < 1:
        raise ValueError("Invalid input")
    progression = int(input('Enter 0/1 for AP/GP: '))
    if progression < 0 or progression > 1 or (progression == 1 and dist_interval == 1):
        raise ValueError("Invalid input")
    return min_dist, year, terms, dist_interval, progression


def get_landuse_pixels(img_mat):

    """
    Extract two classes of the boolean image.
    :param img_mat: Image array read from the GDAL image.
    :return: Tuple containing list of pixel positions of the two classes.
    """

    cols, rows = img_mat.shape
    land = []
    water = []
    for i in range(rows):
        for j in range(cols):
            if img_mat[j, i] == 0:
                land.append((j, i))
            else:
                water.append((j, i))
    return land, water


def generate_cmat(image, min_dist, terms, dist_interval, progression):

    """
    Create C matrix as specified in Graf's model.
    :param image: Image array to be used/
    :param min_dist: Minimum lateral (dl) and upstream distance (du). Must not be less than the pixel size.
    :param terms: Number of rows and columns of the transition matrix.
    :param dist_interval: Interval at which dl and du are to be generated.
    :param progression: Generate (dl, du) as terms of an Arithmetic Progression if set to 0.
           If 1, then Geometric Progression (GP) is used.
    :return: Computed C Matrix as a numpy array.
    """

    pixel_size = int(image.GetGeoTransform()[1])  # get pixel size
    if min_dist < pixel_size:
        raise ValueError("Minimum distance must not be less than pixel size")
    img_mat = image.GetRasterBand(1).ReadAsArray()  # since this is a Boolean image, there exists only 1 band.
    print("Extracting Land and Water Pixels...")
    land, water = get_landuse_pixels(img_mat)  # get list of land and water pixels
    print("Extraction Complete!")
    print("#Land Pixels = ", len(land), "\n#Water Pixels = ", len(water))
    if progression == 0:  # For AP
        dist_interval *= pixel_size
        dist_range = [min_dist + i * dist_interval for i in range(terms)]
    else:  # For GP
        dist_range = [min_dist * dist_interval ** i for i in range(terms)]
    max_dist = dist_range[-1]  # Last term of the series is the highest term.
    print("(dl, du): ", list(dist_range))
    print("Calculating lateral and upstream distances...")
    dist = []
    for l in land:
        for w in water:
            du = np.abs(l[0] - w[0]) * pixel_size
            dl = np.abs(l[1] - w[1]) * pixel_size
            if dl <= max_dist and du <= max_dist:
                dist.append((dl, du))  # add valid land pixels to the list
    print("Distance computation complete!")
    cmat = np.zeros((terms, terms))  # create null matrix of termsXterms size
    dist_dict = {}
    print("Computing C Matrix...")
    for (row, dl) in enumerate(dist_range):
        for (col, du) in enumerate(dist_range):
            for(index, d) in enumerate(dist):
                if d[0] <= dl and d[1] <= du and index not in dist_dict.keys():
                    cmat[row, col] += 1  # add land pixel to the (row, col) position of the C Matrix.
                    # land pixels satisfying the distance condition are added to the dictionary and are not used further
                    dist_dict[index] = True
    print("Finished C Matrix!")
    return cmat


def generate_transition_mat(images, calc_year, min_dist, terms, dist_interval, progression):

    """
    Function to calculate the transition matrix specified in Graf's model.
    :param images: Dictionary of images having year as key and GDAL Image Object as value.
    :param calc_year: User specified year for which the transition matrix is computed.
    :param min_dist: Minimum lateral (dl) and upstream distance (du). Must not be less than the pixel size.
    :param terms: Number of rows and columns of the transition matrix.
    :param dist_interval: Interval at which dl and du are to be generated.
    :param progression: Generate (dl, du) as terms of an Arithmetic Progression if set to 0.
           If 1, then Geometric Progression (GP) is used.
    :return: The transition matrix as a tuple of (C, E, P) where E is the Erosion Matrix and
             P is the Probability Matrix. C, E, P are numpy arrays.
    """

    print("Calculating Transition Matrix...")
    print("\nYEAR " + str(calc_year) + "...")
    calc_year_cmat = generate_cmat(images[calc_year], min_dist, terms, dist_interval, progression)
    final_year = max(list(images.keys()))
    print("\nFINAL YEAR " + str(final_year) + "...")
    final_year_cmat = generate_cmat(images[final_year], min_dist, terms, dist_interval, progression)
    print("\nC" + str(final_year) + "=", final_year_cmat)
    eroded_mat = np.abs(calc_year_cmat - final_year_cmat)
    # the following line simply replaces NaN with zeros in case the denominator is 0.
    pmat = np.divide(eroded_mat, calc_year_cmat, out = np.zeros_like(eroded_mat), where = calc_year_cmat != 0)
    print("\nTransition Matrix Calculated...\n")
    return calc_year_cmat, eroded_mat, pmat


images = read_images('Data', '*.tif')  # get image dictionary
min_dist, year, terms, dist_interval, progression = read_input(images.keys())  # read required inputs
cmat, emat, pmat = generate_transition_mat(images, year, min_dist,
                                           terms, dist_interval, progression)  # get transition matrix

print("C" + str(year) + "= ", cmat)
print("E = ", emat)
print("P = ", pmat)
