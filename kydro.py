from osgeo import gdal
from collections import defaultdict
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
    Enter calculation year: 2006
    Enter end year: 2002 
    Enter number of rows/cols in transition matrix: 5
    Enter distance interval: 2
    Enter 0/1 for AP/GP: 1
    '''
    print("\nEnter the following inputs...")
    min_dist = int(input('Enter minimum distance: '))
    if min_dist < 0:
        raise ValueError('Invalid input')
    calc_year = int(input('Enter calculation year: '))
    if calc_year not in years:
        raise ValueError("Invalid year")
    end_year = int(input('Enter end year: '))
    if end_year not in years:
        raise ValueError("Invalid year")
    terms = int(input('Enter number of rows/cols in transition matrix: '))
    dist_interval = int(input('Enter distance interval: '))
    if terms < 1 or dist_interval < 1:
        raise ValueError("Invalid input")
    progression = int(input('Enter 0/1 for AP/GP: '))
    if progression < 0 or progression > 1 or (progression == 1 and dist_interval == 1):
        raise ValueError("Invalid input")
    return min_dist, calc_year, end_year, terms, dist_interval, progression


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


def get_nearest_channel(land_pixels, water_pixels, pixel_size, max_dist):

    """
    Get nearest channel information
    :param land_pixels: All land pixels
    :param water_pixels: All water pixels
    :param pixel_size: Size of each pixel
    :param max_dist: Maximum allowed dl or du
    :return: Dictionary containing nearest channel information.
    """

    dist_land_dict = {}
    for l in land_pixels:  # for each land pixel
        dl_list = []
        du_list = []
        for w in water_pixels:  # for each water pixel
            du = np.abs(l[0] - w[0]) * pixel_size
            dl = np.abs(l[1] - w[1]) * pixel_size
            if dl <= max_dist and du <= max_dist:  # distances more than max_dist are not considered
                dl_list.append(dl)
                du_list.append(du)
        if dl_list and du_list:  # check if dl and du lists are not empty
            dist_land_dict[l] = (min(dl_list), min(du_list))  # add valid land pixels and respective distances
    return dist_land_dict


def generate_cmat(image, min_dist, terms, dist_interval, progression):

    """
    Create C matrix as specified in Graf's model.
    :param image: Image array to be used
    :param min_dist: Minimum lateral (dl) and upstream distance (du). Must not be less than the pixel size.
    :param terms: Number of rows and columns of the transition matrix.
    :param dist_interval: Interval at which dl and du are to be generated.
    :param progression: Generate (dl, du) as terms of an Arithmetic Progression if set to 0.
           If 1, then Geometric Progression (GP) is used.
    :return: Computed C Matrix as a numpy array along with land, water and selected land pixels as a tuple of tuples.
    """

    pixel_size = int(image.GetGeoTransform()[1])  # get pixel size
    if min_dist < pixel_size:
        raise ValueError("Minimum distance must not be less than pixel size")
    img_mat = image.GetRasterBand(1).ReadAsArray()  # since this is a Boolean image, there exists only 1 band.
    print("Extracting Land and Water Pixels...")
    land_pixels, water_pixels = get_landuse_pixels(img_mat)  # get list of land and water pixels
    print("Extraction Complete!")
    print("#Land Pixels = ", len(land_pixels), "\n#Water Pixels = ", len(water_pixels))
    if progression == 0:  # For AP
        dist_interval *= pixel_size
        dist_range = [min_dist + i * dist_interval for i in range(terms)]
    else:  # For GP
        dist_range = [min_dist * dist_interval ** i for i in range(terms)]
    max_dist = dist_range[-1]  # Last term of the series is the highest term.
    print("(dl, du): ", list(dist_range))
    print("Calculating lateral and upstream distances...")
    dist_land_dict = get_nearest_channel(land_pixels, water_pixels, pixel_size, max_dist)  # get nearest channels
    print("Distance computation complete!")
    cmat = np.zeros((terms, terms))  # create null matrix of termsXterms size
    print("Computing C Matrix...")
    selected_land_pixels = defaultdict(lambda: [])
    dist_land = {}
    for (row, dl) in enumerate(dist_range):
        for (col, du) in enumerate(dist_range):
            for (land, dist) in dist_land_dict.items():
                if dist[0] <= dl and dist[1] <= du and land not in dist_land.keys():  # pixels that are already counted are discarded
                    cmat[row, col] += 1  # add land pixel to the (row, col) position of the C Matrix.
                    # land pixels satisfying the distance condition are added to the dictionary
                    # already selected pixel is discarded
                    selected_land_pixels[(dl, du)].append(land)
                    dist_land[land] = True
    print("Finished C Matrix!")
    return cmat, (land_pixels, water_pixels, selected_land_pixels, img_mat.shape)


def generate_transition_mat(images, calc_year, final_year, min_dist, terms, dist_interval, progression):

    """
    Function to calculate the transition matrix specified in Graf's model.
    :param images: Dictionary of images having year as key and GDAL Image Object as value.
    :param calc_year: User specified year for which the transition matrix is computed.
    :param final_year: User specified final year using which erosion matrix is computed.
    :param min_dist: Minimum lateral (dl) and upstream distance (du). Must not be less than the pixel size.
    :param terms: Number of rows and columns of the transition matrix.
    :param dist_interval: Interval at which dl and du are to be generated.
    :param progression: Generate (dl, du) as terms of an Arithmetic Progression if set to 0.
           If 1, then Geometric Progression (GP) is used.
    :return: The transition matrix and map information as a tuple of (C, E, P, map_dict)
            where E is the Erosion Matrix and P is the Probability Matrix.
            C, E, P are numpy arrays and map_dict is a dictionary of map information.
    """

    print("Calculating Transition Matrix...")
    print("\nYEAR " + str(calc_year) + "...")
    map_dict = {}
    calc_year_cmat, map_dict[calc_year] = generate_cmat(images[calc_year], min_dist, terms, dist_interval, progression)
    print("\nFINAL YEAR " + str(final_year) + "...")
    final_year_cmat, map_dict[final_year] = generate_cmat(images[final_year], min_dist, terms, dist_interval, progression)
    print("\nC" + str(final_year) + "=", final_year_cmat)
    eroded_mat = calc_year_cmat - final_year_cmat
    # the following line simply replaces NaN with zeros in case the denominator is 0.
    pmat = np.divide(eroded_mat, calc_year_cmat, out = np.zeros_like(eroded_mat), where = calc_year_cmat != 0)
    print("\nTransition Matrix Calculated...\n")
    return calc_year_cmat, eroded_mat, pmat, map_dict


def set_pixels(img_size, pixel_list, exclusion_list = ()):

    """
    Set pixel values of a new 8-bit image
    :param img_size: Shape of the image in (col,row) format
    :param pixel_list: Pixels of a specific class to assign value.
    :param exclusion_list: Do not assign pixel values for items in this list.
    :return: new image array
    """

    cols, rows = img_size
    arr = np.zeros(img_size)
    for i in range(rows):
        for j in range(cols):
            if (j, i) in pixel_list and (j, i) not in exclusion_list:
                arr[(j, i)] = 255
    return arr


def create_maps(map_info, projection, geotransform, outfile):

    """
    Create RGB maps for land and water classes
    :param geotransform: Get input raster parameters
    :param projection: Get input raster projection
    :param map_info: highlighting land, water and land pixels belonging to specified distance classes
    :param outfile: output file name without extension
    :return: None
    """

    print('\nCreating RGB maps...')
    if not os.path.exists('Maps'):
        os.mkdir('Maps')
    land, water, slp, img_size = map_info
    bands = [set_pixels(img_size, water)]
    driver = gdal.GetDriverByName("GTiff")
    outfile = 'Maps/' + outfile  # store generated maps in Maps directory
    print(slp)
    for dist in slp.keys():
        bands.append(set_pixels(img_size, land, slp[dist]))
        bands.append(set_pixels(img_size, slp[dist]))
        new_file = outfile + "_" + str(dist[0]) + "_" + str(dist[1]) + '.tif'
        outdata = driver.Create(new_file, img_size[1], img_size[0], 3, gdal.GDT_Byte)  # set 3 bands, 24-bit image
        outdata.SetProjection(projection)
        outdata.SetGeoTransform(geotransform)
        for (index, band) in enumerate(bands):
            outdata.GetRasterBand(index + 1).WriteArray(band)
        outdata.FlushCache()
        del bands[1:]
    print('All Maps created!')


images = read_images('Data', '*.tif')  # get image dictionary

min_dist, calc_year, end_year, terms, dist_interval, progression = read_input(images.keys())  # read required inputs
cmat, emat, pmat, map_dict = generate_transition_mat(images, calc_year, end_year, min_dist,
                                           terms, dist_interval, progression)  # get transition matrix

print("C" + str(calc_year) + "= ", cmat)
print("E = ", emat)
print("P = ", pmat)

create_maps(map_dict[calc_year], images[calc_year].GetProjection(), images[calc_year].GetGeoTransform(), 'map')
