import argparse
import sys
import cv2
import numpy as np
from head import *
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--show",  action="store_true", help="show the video while processing.")
parser.add_argument("-d", "--delay", type=int,            help="set the delay between frames (ms)", default=1)
parser.add_argument("video",         type=str,            help="input video file")
parser.add_argument("prj",           type=str,            help="project info file")
parser.add_argument("mask",          type=str,            help="mask coords (<left>x<top>:<width>x<height>)")
parser.add_argument("lumth",         type=int,            help="Luminosity threshold (ex. 200)")
args = parser.parse_args()
filename = 'noFish1.png'
img = cv2.imread(filename)
k=0
p= np.empty(5)
q= np.empty(5)
pg = np.empty(6)
FILE_OUTPUT2 = 'outTrack.avi'
if os.path.isfile(FILE_OUTPUT2):
    os.remove(FILE_OUTPUT2)
fourcc = cv2.cv.CV_FOURCC(*'MJPG')
out2 = cv2.VideoWriter(FILE_OUTPUT2,fourcc, 20.0, (int(640),int(480)),0)
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
	global k
	p[k]=int(x)
	q[k]=int(y)
        cv2.circle(img,(x,y),3,(0,255,0),-1)
	k=k+1
# Create a black image, a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(k<5):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27 :
        break
pg = [int(p[0]),int(q[0]),int(p[1]),int(q[1]),int(p[2]),int(q[2]),int(p[3]),int(q[3])]
#s1=(pg[0]-pg[6])(pg[0]-pg[6])
#s2=(pg[1]-pg[7])(pg[1]-pg[7])
#sd=math.sqrt(s1 + s2)
#sr=0.15/sd
# open the video file
capture = cv2.VideoCapture(args.video)
frame_height = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
frame_width  = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
frame_count  = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
# create the square mask
(mx, my, mw, mh) = parse_mask(args.mask)
mask = np.uint8(np.zeros((int(frame_height), int(frame_width))))
mask[my:my + mh, mx:mx + mw] = 255

# create the project file
prj = Project()
prj.set("video", args.video)
prj.set("mask", (mx, my, mw, mh))
prj.set("lumth", args.lumth)
prj.save(args.prj)

# open the raw data file
fraw = open(prj.get_raw_fname(), "w")

# start the counter
tcount = TimeCount(frame_count)

if args.show:
    sys.stderr.write("\nPress 'Q' or 'q' to terminate.\n")

last_head = None
last_tail = None
c=0
m=0
f = -1
xp = 0
yp = 0
tp = -1
velocity = 0
centroid =0
while True:
    f += 1

    (ret, frame) = capture.read()

    if(not ret):
        break

    if(f % 100 == 0):
        tcount.show(f)
    fr=frame.copy();
    frame = cv2.absdiff(frame,img)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    [hue, sat, lum] = cv2.split(hsv)

    (ret, lum_bin) = cv2.threshold(lum, args.lumth, 255, cv2.THRESH_BINARY)

    #lum_bin = np.bitwise_and(lum_bin, mask)

    if args.show:
        cv2.imshow("Tracking Binary", lum_bin)

    (blobs, dummy) = cv2.findContours(lum_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    blobs = sorted(blobs, key=lambda x: -len(x))

    body_length = 0
    txt = ""

    if (len(blobs) > 0) and (np.size(blobs[0]) > 100):
        blob = blobs[0]

        small_mask = np.uint8(np.ones(np.shape(frame)[:2])) * 0
        cv2.fillConvexPoly(small_mask, blob, 255)

        moments = cv2.moments(small_mask)
        centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        dists = map(lambda p: lindist(p[0], centroid), blob)
        tail = tuple(blob[dists.index(max(dists))][0])

        dists = map(lambda p: lindist(p[0], tail), blob)
        head = tuple(blob[dists.index(max(dists))][0])

        # doesn't consider when the fish touches the limits
        if(check_inside(head[0], mx, mw) and check_inside(head[1], my, mh) and check_inside(tail[0], mx, mw) and check_inside(tail[1], my, mh)):
            body_angle = angle(head, centroid, tail)

            # swap the head and the tail when needed
            if (last_head is not None) and (lindist(head, last_head) > lindist(head, last_tail)) and (body_angle > 20):
                (head, tail) = (tail, head)
        txt = "%d\t1\t%d\t%d\t%d\t%d\t%d\t%d\n" % (f, head[0], head[1], centroid[0], centroid[1], tail[0], tail[1])

        # store the head and tail for the next frame
        last_head = head
        last_tail = tail
    if args.show:
        cv2.putText(frame, "%d" % f, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        #cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)
        cv2.putText(frame, "%dx%d:%dx%d (%d)" % (mx, my, mw, mh, args.lumth), (mx, my+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if(txt == ""):
            cv2.line(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 1)
            cv2.line(frame, (mx, my + mh), (mx + mw, my), (0, 255, 0), 1)
        else:
            cv2.circle(fr, centroid, 2, (0, 255, 0), -1)
            cv2.circle(fr, tail, 2, (255, 0, 0), -1)
            cv2.circle(fr, head, 2, (0, 0, 255), -1)
            cv2.line(fr, centroid, head, (0, 255, 0), 1)
	    cv2.line(fr,last_tail,tail,(0,255,255),1)
	    #pretail=tail

    if txt == "":
        last_head = None
        last_tail = None
        txt = "%d\t0\t0\t0\t0\t0\t0\t0\n" % f

    fraw.write(txt)
    fraw.flush()

    if args.show:
	#pg=np.array(polygon,np.int32)
	#pg = pg.reshape((-1,1,2))
	#frame = cv2.polylines(frame, (pg[0],pg[1]),(pg[2],pg[3]),(pg[4],pg[5]),(pg[6],pg[7]) , True, (0,255,255),3)
	cv2.line(fr, (pg[0],pg[1]),(pg[2],pg[3]), (0, 255, 0), 2)
	cv2.line(fr, (pg[2],pg[3]),(pg[4],pg[5]), (0, 255, 0), 2)
	cv2.line(fr, (pg[4],pg[5]),(pg[6],pg[7]), (0, 255, 0), 2)
	cv2.line(fr, (pg[6],pg[7]),(pg[0],pg[1]), (0, 255, 0), 2)
	cv2.line(fr, ((pg[0]+pg[6])/2,(pg[1]+pg[7])/2),((pg[2]+pg[4])/2,(pg[3]+pg[5])/2), (255, 0, 0), 1)
        cv2.putText(fr, "%d" % f, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
	#cv2.line(fr, (last_tail[0],last_tail[1]),(tail[0],tail[1]), (0,0,255), 1)
        cv2.imshow("Tracking Video", fr)
	#fr =cv2.cvtColor(fr,cv2.COLOR_GRAY2BGR)
	fr1=np.array(fr)
	out2.write(fr1)
        key = cv2.waitKey(args.delay)
        if key > 0:
            if key in QUIT_KEYS:
                sys.stderr.write("\n\nFinal parameters: %dx%d:%dx%d %d\n" % (mx, my, mw, mh, args.lumth))
                break

            if key in [LUMTH_UP, LUMTH_DOWN]:
                args.lumth = max(0, args.lumth - 1) if key == LUMTH_DOWN else min(args.lumth + 1, 255)

            if key in MOVE_SQUARE_KEYS.keys():
                mx += MOVE_SQUARE_KEYS[key][0]
                my += MOVE_SQUARE_KEYS[key][1]
                mw += MOVE_SQUARE_KEYS[key][2]
                mh += MOVE_SQUARE_KEYS[key][3]

                mask[:, :] = 0
                mask[my:my + mh, mx:mx + mw] = 255
out2.release()
fraw.close()
sys.stderr.write("\nDONE\n")
