from osgeo import gdal
import numpy as np
import glob
import os

def read_images(path, imgformat):
    images = {}
    files = os.path.join(path, imgformat)
    for file in glob.glob(files):
        pos = file.rfind(os.sep) + 1
        year = int(file[pos : pos + 4])
        images[year] = gdal.Open(file)
    return images

def get_landuse_pixels(img_mat):
    cols, rows = img_mat.shape
    land = []
    water = []
    for i in range(rows):
        for j in range(cols):
            if img_mat[j, i] == 0:
                land.append((j,i))
            else:
                water.append((j,i))
    return land, water

def generate_cmat(image, min_dist, max_dist, dist_interval = 2):
    pixel_size = int(image.GetGeoTransform()[1])
    if min_dist < pixel_size:
        raise ValueError("Minimum distance must not be less than pixel size")
    img_mat = image.GetRasterBand(1).ReadAsArray()
    print("Extracting Land and Water Pixels...")
    land, water = get_landuse_pixels(img_mat)
    print("Extraction Complete!")
    print("#Land Pixels = ", len(land), "\n#Water Pixels = ", len(water))
    dist = []
    print("Calculating lateral and upstream distances...")
    for l in land:
        for w in water:
            du = np.abs(l[0] - w[0]) * pixel_size
            dl = np.abs(l[1] - w[1]) * pixel_size
            if dl <= max_dist and du <= max_dist:
                dist.append((dl,du))

    print("Distance computation complete!")
    dist_interval *= pixel_size
    dist_range = range(min_dist, max_dist + 1, dist_interval)
    n = len(dist_range)
    cmat = np.zeros((n,n))
    dist_dict = {}
    print("Computing C Matrix...")
    for (row, dl) in enumerate(dist_range):
        for (col, du) in enumerate(dist_range):
            i = 0
            while i < len(dist):
                d = dist[i]
                if d[0] <= dl and d[1] <= du and i not in dist_dict.keys():
                    cmat[row, col] += 1
                    dist_dict[i] = True
                i += 1
    print("Finished C Matrix!")
    return cmat

def generate_transition_mat(images, calc_year, min_dist, max_dist, dist_interval):
    if min_dist > max_dist:
        raise ValueError("Invalid input")
    print("\nYEAR " + str(calc_year) + "...")
    calc_year_cmat = generate_cmat(images[calc_year], min_dist, max_dist, dist_interval)
    final_year = max(list(images.keys()))
    print("\nFINAL YEAR " + str(final_year) + "...")
    final_year_cmat = generate_cmat(images[final_year], min_dist, max_dist, dist_interval)
    eroded_mat = np.abs(calc_year_cmat - final_year_cmat)
    pmat = np.divide(eroded_mat, calc_year_cmat, out = np.zeros_like(eroded_mat), where = calc_year_cmat != 0)
    return calc_year_cmat, eroded_mat, pmat

print("Reading images...")
images = read_images('Data','*.tif')
print("Finished reading")
print("Calculating Transition Matrix...")
cmat, emat, pmat = generate_transition_mat(images, 2010, 50, 200, 1)
print("Transition Matrix Calculated...\n")
print("C = ", cmat)
print("E = ", emat)
print("P = ", pmat)

