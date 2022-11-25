import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects



def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled

def fill_label_holes_cv2(lbl_img):
    import cv2
    h, w = lbl_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    filled = cv2.floodFill(lbl_img, None, (0, 0), int(lbl_img.max()));

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(filled[2])

    # Combine the two images to get the foreground.
    #im_out = lbl_img | im_floodfill_inv
    #print(f'{lbl_img.shape=},\n,{lbl_img.max()=}, \n,{lbl_img.min()=}')
    #print(f'{im_out.shape=},\n,{im_out.max()=}, \n,{im_out.min()=}')
    height, width = im_floodfill_inv.shape
    im_floodfill_inv = np.where(im_floodfill_inv==254, 0, lbl_img.max())
    return im_floodfill_inv[1:height-1, 1:width-1]