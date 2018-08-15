import cv2
import numpy as np
from collections import namedtuple
import pytesseract
import imutils
import collections
from matplotlib import pyplot as plt

def crop_frame_picture(image):
    #img = image[:, :, 0]
    #height,width = img.shape
    height, width,_ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            if((approx[2][0][0] - approx[0][0][0]) >= ((width/100)*60) and (approx[2][0][1] - approx[0][0][1]) >= ((height/100)*60)):
                x1 = approx[0][0][0] + 7
                y1 = approx[0][0][1] + 7
                x2 = approx[1][0][0] + 7
                y2 = approx[1][0][1] - 7
                x3 = approx[2][0][0] - 7
                y3 = approx[2][0][1] - 7
                x4 = approx[3][0][0] - 7
                y4 = approx[3][0][1] + 7
                screenCnt = np.array([[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]])
                image = image[screenCnt[0][0][1]:screenCnt[2][0][1], screenCnt[0][0][0]:screenCnt[2][0][0]]
                return image
            else:
                return image
        else:
            return image

def repeatly(data):
    c = collections.Counter(data)
    c = c.most_common(3)
    #print(c)
    c_max = c[0]
    c_max_data = c[0][1]
    for i in range(len(c)):
        if (c_max_data < c[i][1]):
            c_max = c[i]
            c_max_data = c[i][1]
    return c_max[0]

def area_mser(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return None
    else:
        return b.xmin,b.ymin,b.xmax,b.ymax

def mean_mser(data):
    count = 1
    all_data = 0
    for i in data:
        all_data += i
        count += 1
    avr = all_data/count
    return  avr

def tesseract_seg(image,array = np.array([])):
    h, w, _ = image.shape
    boxes = pytesseract.image_to_boxes(image, lang='tha')
    num_count = 0
    x_d = []
    y_d = []
    w_d = []
    h_d = []
    for b in boxes.splitlines():
        b = b.split(' ')
        #cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)
        xmin = int(b[1])
        ymax = h - int(b[2])
        xmax = int(b[3])
        ymin = h - int(b[4])
        x_d.append(xmin)
        y_d.append(ymin)
        w_d.append(xmax - xmin)
        h_d.append(ymax - ymin)
        num_count += 1
    if(len(w_d) >= 3 and len(h_d) >= 3):
        m_w, m_h = repeatly(w_d), repeatly(h_d)
        for i in range(num_count):
            data = x_d[i],y_d[i], x_d[i]+w_d[i], y_d[i] + h_d[i]
            if (((x_d[i] + w_d[i]) - x_d[i]) <= (m_w * 3) and ((y_d[i] + h_d[i]) - y_d[i]) <= (m_h * 1.4)):
                if len(array)  == 0:
                    array = np.array([data])
                else:
                    sub = np.array(data)
                    array = np.vstack((sub, array))
    else:
        for i in range(num_count):
            data = x_d[i],y_d[i], x_d[i]+w_d[i], y_d[i] + h_d[i]
            if len(array) == 0:
                array = np.array([data])
            else:
                sub = np.array(data)
                array = np.vstack((sub, array))
    return array

def ocr_findcontour(image,array = np.array([])):
    gray_each = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, thresh_each = cv2.threshold(gray_each, 127, 255, cv2.THRESH_BINARY_INV)
    im2, ctrs2, hier2 = cv2.findContours(thresh_each.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_count = 0
    x_d = []
    y_d = []
    w_d = []
    h_d = []
    for j, c in enumerate(ctrs2):
        x, y, w, h = cv2.boundingRect(c)
        x_d.append(x)
        y_d.append(y)
        w_d.append(w)
        h_d.append(h)
        num_count += 1
    if (len(w_d) >= 3 and len(h_d) >= 3):
        m_w, m_h = repeatly(w_d), repeatly(h_d)
        for i in range(num_count):
            data = x_d[i],y_d[i], x_d[i]+w_d[i], y_d[i] + h_d[i]
            if (((x_d[i]+w_d[i]) - x_d[i]) <= (m_w * 5) and ((y_d[i] + h_d[i]) - y_d[i]) <= (m_h * 2)):
                if len(array)  == 0:
                    array = np.array([data])
                else:
                    sub = np.array(data)
                    array = np.vstack((sub, array))
    else:
        for i in range(num_count):
            data = x_d[i],y_d[i], x_d[i]+w_d[i], y_d[i] + h_d[i]
            if len(array) == 0:
                array = np.array([data])
            else:
                sub = np.array(data)
                array = np.vstack((sub, array))
    return array

def ocr_mser(image,array = np.array([])):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    num_count = 0
    x_d = []
    y_d = []
    w_d = []
    h_d = []
    for i, c in enumerate(hulls):
        x, y, w, h = cv2.boundingRect(c)
        x_d.append(x)
        y_d.append(y)
        w_d.append(w)
        h_d.append(h)
        num_count += 1
    if (len(w_d) >= 3 and len(h_d) >= 3):
        m_w, m_h = repeatly(w_d), repeatly(h_d)
        mean_w, mean_h = mean_mser(w_d),mean_mser(h_d)
        for i in range(num_count):
            if(len(hulls) > 1):
                if i > 1:
                    a = Rectangle(x_d[i - 1], y_d[i - 1], x_d[i - 1] + w_d[i - 1], y_d[i - 1] + h_d[i - 1])
                    b = Rectangle(x_d[i], y_d[i], x_d[i] + w_d[i], y_d[i] + h_d[i])
                    if area_mser(a, b) != None:
                        x, y, xm, ym = area_mser(a, b)
                        if ((xm - x) <= (mean_w * 0.2) and (ym - y) <= (m_h * 2)):
                            data = x, y, xm, ym
                            if len(array) == 0:
                                array = np.array([data])
                            else:
                                sub = np.array(data)
                                array = np.vstack((sub, array))
            else:
                data = x_d[i], y_d[i], x_d[i] + w_d[i], y_d[i] + h_d[i]
                array = np.array([data])
    else:
        for i in range(num_count):
            if(len(hulls) > 1):
                if i > 1:
                    a = Rectangle(x_d[i - 1], y_d[i - 1], x_d[i - 1] + w_d[i - 1], y_d[i - 1] + h_d[i - 1])
                    b = Rectangle(x_d[i], y_d[i], x_d[i] + w_d[i], y_d[i] + h_d[i])
                    if area_mser(a, b) != None:
                        x, y, xm, ym = area_mser(a, b)
                        data = x, y, xm, ym
                        if len(array) == 0:
                            array = np.array([data])
                        else:
                            sub = np.array(data)
                            array = np.vstack((sub, array))
            else:
                data = x_d[i], y_d[i], x_d[i] + w_d[i], y_d[i] + h_d[i]
                array = np.array([data])

    return  array

def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick]

def get_data_list_xmin(data):
    return data[0]

def get_data_list_ymin(data):
    return data[1]

def cha_segment(image):
    image = crop_frame_picture(image)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)

    #dilation (make line)
    kernel = np.ones((15,120), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    im,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    image_list = []
    data_space = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        data_space.append((x,y,w+x,h+y))
    new_data_space = sorted(data_space, key= get_data_list_ymin)
    for i in range(len(new_data_space)):
        if i < (len(new_data_space)-1):
            if new_data_space[i][1] == new_data_space[i+1][1]:
                if new_data_space[i][0] <= new_data_space[i+1][0]:
                    x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2],new_data_space[i][3]
                    roi = image[y:ymax, x:xmax]
                    image_list.append(roi)
                else:
                    keep_data = new_data_space[i]
                    new_data_space[i] = new_data_space[i+1]
                    new_data_space[i+1] = keep_data
                    x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2],new_data_space[i][3]
                    roi = image[y:ymax, x:xmax]
                    image_list.append(roi)
            else:
                    x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], new_data_space[i][3]
                    roi = image[y:ymax, x:xmax]
                    image_list.append(roi)
        else:
            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], new_data_space[i][3]
            roi = image[y:ymax, x:xmax]
            image_list.append(roi)

    lst_bnd = []
    for img in image_list:
        bnd = tesseract_seg(img)
        bnd = ocr_findcontour(img,bnd)
        #bnd = ocr_findcontour(img)
        bnd = ocr_mser(img,bnd)
        #bnd = ocr_mser(img)
        img_bnd = [(img, bnd)]
        for (imagePath, boundingBoxes) in img_bnd:
            sub_bnd = []
            print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
            image_nms = imagePath
            orig = image.copy()
            pick = non_max_suppression_slow(boundingBoxes, 0.3)
            new_pick = np.array([])
            new_data_space = sorted(pick, key=get_data_list_xmin)
            for i in range(len(new_data_space)):
                if i < (len(new_data_space) - 2):
                    if new_data_space[i][2] - new_data_space[i+1][0] < (((new_data_space[i][2] - new_data_space[i][0])/100)*35) and new_data_space[i][2] - new_data_space[i+1][0] > 1 and new_data_space[i+1][2] != new_data_space[i][2]:
                        if new_data_space[i+1][1] < new_data_space[i][1]:
                            if new_data_space[i+1][1] < new_data_space[i+2][1]:
                                keep_data = new_data_space[i + 1]
                                new_data_space[i + 1] = new_data_space[i + 2]
                                new_data_space[i + 2] = keep_data
                                x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                                   new_data_space[i][3]
                            else:
                                x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], new_data_space[i][3]
                        else:
                            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                               new_data_space[i][3]
                    elif new_data_space[i][0] == new_data_space[i+1][0]:
                        if(new_data_space[i][1] >= new_data_space[i+1][1]):
                            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                               new_data_space[i][3]
                        else:
                            keep_data = new_data_space[i]
                            new_data_space[i] = new_data_space[i + 1]
                            new_data_space[i + 1] = keep_data
                            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                               new_data_space[i][3]
                    else:
                        x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                           new_data_space[i][3]
                    if len(new_pick) == 0:
                        new_pick = np.array([(x, y, xmax, ymax)])
                    else:
                        sub = np.array((x, y, xmax, ymax))
                        new_pick = np.vstack((new_pick, sub))
                elif i == (len(new_data_space) - 2):
                    if new_data_space[i][2] - new_data_space[i + 1][0] < (
                        ((new_data_space[i][2] - new_data_space[i][0]) / 100) * 35) and new_data_space[i][2] - \
                            new_data_space[i + 1][0] > 1 and new_data_space[i + 1][2] != new_data_space[i][2]:
                        if new_data_space[i][1] >= new_data_space[i+1][1]:
                            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                               new_data_space[i][3]
                        else:
                            keep_data = new_data_space[i]
                            new_data_space[i] = new_data_space[i + 1]
                            new_data_space[i + 1] = keep_data
                            x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                               new_data_space[i][3]
                    else:
                        x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                           new_data_space[i][3]
                    if len(new_pick) == 0:
                        new_pick = np.array([(x, y, xmax, ymax)])
                    else:
                        sub = np.array((x, y, xmax, ymax))
                        new_pick = np.vstack((new_pick, sub))
                else:
                    x, y, xmax, ymax = new_data_space[i][0], new_data_space[i][1], new_data_space[i][2], \
                                       new_data_space[i][3]
                    if len(new_pick) == 0:
                        new_pick = np.array([(x, y, xmax, ymax)])
                    else:
                        sub = np.array((x, y, xmax, ymax))
                        new_pick = np.vstack((new_pick, sub))
            print("[x] after applying non-maximum, %d bounding boxes" % (len(new_pick)))


            for (startX, startY, endX, endY) in new_pick:
                data = startX, startY, endX, endY
                sub_bnd.append(data)
                cv2.rectangle(image_nms, (startX, startY), (endX, endY), (0, 255, 0), 1)
                #cv2.imshow('marked areas',img)
                #cv2.waitKey(0)
            lst_bnd.append(sub_bnd)
        #cv2.namedWindow('marked areas',0)
        #cv2.imshow('marked areas',img)
        #cv2.waitKey(0)
    return image_list,lst_bnd

if __name__ == "__main__":
    path_image = "image/t.jpg"
    image = cv2.imread(path_image)
    img ,data_list = cha_segment(image)
    num = 0
    for j in img:
        for data in data_list[num]:
            pass
            #print(data[0],data[1],data[2],data[3])
        cv2.imshow('marked areas', j)
        cv2.waitKey(0)
        num += 1