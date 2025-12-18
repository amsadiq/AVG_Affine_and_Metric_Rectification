import numpy as np
import cv2
from matplotlib import pyplot as plt


def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

filename = '/Users/sadiq/Downloads/AVG lab 1/AVG lab 1/fig1_6c__.jpg'
img_bgr = cv2.imread(filename)
if img_bgr is None:
    raise IOError(f"Cannot read image at {filename}")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img)
pts = np.asarray(plt.ginput(4, timeout=-1))
plt.show()

print('chosen coord: ', pts)  # each row is a point

if len(pts) > 0:
    plt.plot(*zip(*pts), marker='o', color='r', ls='')
    plt.imshow(img)
    plt.show()

# Use test data if no points were clicked
if len(pts) == 0:
    points = np.array([
        [1053.01298701,  460.10606061],
        [1364.7012987,   641.92424242],
        [1057.34199134,  867.03246753],
        [ 745.65367965,  641.92424242]
    ])
    print("Using test data points...")
else:
    points = pts.astype(float)   # <— use the clicked points

'''
Affine rectification
'''
print('\n-------- Task 1: Affine rectification --------')
pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)  # convert chosen pts to homogeneous coordinate
x0, x1, x2, x3 = pts_homo
print('Task 1.1: Identify image of the line at inf on projective plane')
hor_0 = np.cross(x0, x1) # line connecting x0 and x1 - the first horizontal line
hor_1 = np.cross(x3, x2) # line connecting x3 and x2 - the 2nd horizontal line
pt_ideal_0 = np.cross(hor_0, hor_1) # Intersection of hor_0 and hor_1 - first ideal point
pt_ideal_0 /= pt_ideal_0[-1]  # normalize
print('@Task 1.1: first ideal point: ', pt_ideal_0)

ver_0 = np.cross(x0, x3) # line connecting x0 and x3 - the 1st vertical line
ver_1 = np.cross(x1, x2) # line connecting x1 and x2 - the 2nd vertical line
pt_ideal_1 = np.cross(ver_0, ver_1) # Intersection of ver_0 and ver_1 - 2nd ideal point
pt_ideal_1 /= pt_ideal_1[-1]
print('@Task 1.1: second ideal point: ', pt_ideal_1)

l_inf = np.cross(pt_ideal_0, pt_ideal_1) # image of line at inf
l_inf /= l_inf[-1]
print('@Task1.1: line at infinity: ', l_inf)
l0, l1, l2 = l_inf 
print('Task 1.2: Construct the projectivity that affinely rectify image')
H = np.array([ 
	[1, 0, 0],
    [0, 1, 0],
    [l0, l1, l2]

]) # Hp' is the H matrix from the instruction document
print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

H_E = euclidean_trans(np.deg2rad(0), 50, 250)

affine_img = cv2.warpPerspective(img, H_E @ H, (img.shape[1], img.shape[0]))

affine_pts = (H_E @ H @ pts_homo.T).T # TODO
for i in range(affine_pts.shape[0]):
    affine_pts[i] /= affine_pts[i, -1]

plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(affine_img)
plt.show()
print('-------- End of Task 1 --------\n')

'''
Task 2: Metric rectification
'''
print('\n-------- Task 2: Metric rectification --------')
print('Task 2.1: transform 4 chosen points from projective image to affine image')

aff_hor_0 = np.cross(affine_pts[0], affine_pts[1]) # image of first horizontal line on affine plane
aff_hor_1 = np.cross(affine_pts[2], affine_pts[3]) # image of 2nd horizontal line on affine plane

aff_ver_0 = np.cross(affine_pts[0], affine_pts[3]) # image of first vertical line on affine plane
aff_ver_1 = np.cross(affine_pts[1], affine_pts[2]) # image of 2nd vertical line on affine plane

# Normalize lines (for numerical stability)
def norm_line(l):
    return l / np.linalg.norm(l[:2])

aff_hor_0 = norm_line(aff_hor_0)
aff_hor_1 = norm_line(aff_hor_1)
aff_ver_0 = norm_line(aff_ver_0)
aff_ver_1 = norm_line(aff_ver_1)

print('@Task 2.1: first chosen point coordinate')
print('\t\t on projective image: ', pts_homo[0])
print('\t\t on affine image: ', affine_pts[0])

print('Task 2.2: construct constraint matrix C to find vector s')
m0, m1, m2 = aff_hor_0
n0, n1, n2 = aff_ver_0

m_0, m_1, m_2 = aff_hor_1
n_0, n_1, n_2 = aff_ver_1

C0 = np.array([m0*n0, m0*n1 + m1*n0, m1*n1])

C1 = np.array([m_0*n_0, m_0*n_1 + m_1*n_0, m_1*n_1])

C = np.vstack([C0, C1])
print('@Task 2.2: constraint matrix C:\n', C)

print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
U, S_vals, V_T = np.linalg.svd(C)
s = V_T.T[:, -1]

print('@Task 2.3: s = ', s)
print('@Task 2.3: C @ s = \n', C @ s.reshape(-1, 1))
mat_S = np.array([
    [s[0], s[1]],
    [s[1], s[2]],
])

print('@Task 2.3: matrix S:\n', mat_S)

print('Task 2.4: Find the projectivity that do metric rectificaiton')
eigenvals, eigenvecs = np.linalg.eig(mat_S)
print('Eigenvalues of S:', eigenvals)

eigenvals, eigenvecs = np.linalg.eig(mat_S)

Lambda_abs = np.diag(np.abs(eigenvals)) # Project S to PSD
mat_S_psd = eigenvecs @ Lambda_abs @ eigenvecs.T
s_psd = np.array([mat_S_psd[0,0], mat_S_psd[0,1], mat_S_psd[1,1]]) #recompute s from this PSD S

# Use mat_S_psd for K factorization
eigenvals_psd, eigenvecs_psd = np.linalg.eig(mat_S_psd)
sqrt_S = np.diag(np.sqrt(eigenvals_psd))
K = eigenvecs_psd @ sqrt_S
if np.linalg.det(K) < 0:
    K = -K
K_inv = np.linalg.inv(K)

H = np.array([
    [K_inv[0, 0], K_inv[0, 1], 0],
    [K_inv[1, 0], K_inv[1, 1], 0],
    [0,          0,          1]
])

aff_dual_conic = np.array([
    [s_psd[0], s_psd[1], 0],
    [s_psd[1], s_psd[2], 0],
    [0,       0,       0]
])

print('@Task 2.3: image of dual conic on metric rectified image: ', H @ aff_dual_conic @ H.T)

H_E = euclidean_trans(np.deg2rad(0), 30, 80)
H_fin = H_E @ H
eucl_img = cv2.warpPerspective(affine_img, H_fin, (img.shape[1], img.shape[0]))

eucl_pts = (H_fin @ affine_pts.T).T
for i in range(eucl_pts.shape[0]):
    eucl_pts[i] /= eucl_pts[i, -1]

plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(eucl_img)
plt.show()
