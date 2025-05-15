#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
from mr_voice.msg import Voice
from pcms.openvino_models import HumanPoseEstimation
from RobotChassis import RobotChassis


# === 狀態定義 ===
class State:
   WAIT_FOR_HUMAN = 0
   ASK_NAME = 1
   WAIT_FOR_REPLY = 2
   FOLLOW_PERSON = 3
   TASK_DONE = 4
   TAKE_BAG = 5
   WAIT_FOR_FOLLOW = 6
   WAIT_FOR_REPLY1 = 7
   TURN_TO_THE_BAG = 8
   WAIT_FOR_BAG = 9
   Navigation = 10
   call_back_home = 11
   back_home = 12
# === 全域變數 ===
_image = None
_depth = None
target_box = None
target_depth = None
target_center = None
voice_text = ""
voice_direction = 0
last_voice_time = 0
asked_bag = False
turn_start_time = None
hund = ""
nameList = ["adam","axel","chris","hunter","jack","max","paris","olivia","william"]




# === 回呼函數 ===
def callback_image(msg):
   global _image
   _image = CvBridge().imgmsg_to_cv2(msg, 'bgr8')


def callback_depth(msg):
   global _depth
   _depth = CvBridge().imgmsg_to_cv2(msg)


def callback_voice(msg):
   global voice_text, voice_direction, last_voice_time
   voice_text = msg.text
   voice_direction = msg.direction
   last_voice_time = rospy.get_time()


def get_real_xyz(dp, x, y):
   a = 49.5 * np.pi / 180
   b = 60.0 * np.pi / 180
   d = dp[y][x]
   h, w = dp.shape[:2]
   x = int(x) - int(w // 2)
   y = int(y) - int(h // 2)
   real_y = round(y * 2 * d * np.tan(a / 2) / h)
   real_x = round(x * 2 * d * np.tan(b / 2) / w)
   return real_x, real_y, d


def is_pointing_to_box(finger, box):
   box_center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
   distance = np.linalg.norm(np.array(finger) - np.array(box_center))
   return distance






# === 主程式 ===
if __name__ == "__main__":
   rospy.init_node("fsm_pose_trackin")
   rospy.loginfo("FSM Pose Trackin Node Started")


   # 訂閱
   _image = None
   _depth = None
   voice_text, voice_direction, last_voice_time = "", 0, 0
   rospy.Subscriber('/camera/rgb/image_raw', Image, callback_image)
   rospy.Subscriber('/camera/depth/image_raw', Image, callback_depth)
   rospy.Subscriber('/voice/text', Voice, callback_voice)


   # 發佈
   pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
   pub_say = rospy.Publisher('/speaker/say', String, queue_size=10, latch=True)
   cmd = Twist()


   # 模型
   pose_model = HumanPoseEstimation()
   model = YOLO("/home/pcms/Downloads/green_chair.pt")


   move_base = RobotChassis()


   rate = rospy.Rate(10)
   rospy.sleep(1)


   # 初始狀態
   state = State.WAIT_FOR_HUMAN
   target_found_once = False
   wait_start_time = rospy.get_time()
  


   while not rospy.is_shutdown():
       if _image is None or _depth is None:
           rate.sleep()
           continue


       frame = _image.copy()
       depth = _depth.copy()


       poses = pose_model.forward(frame)
       #rospy.loginfo("%s (%d)" % (voice_text, voice_direction)){}
       print(poses)
       if poses is not None:
           #frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
           tx, ty, td = -1, -1, -1
           mx, my, md = -1, -1, -1
           ux, uy, ud = -1, -1, -1
           for pose in poses:
               x8, y8, c8 = map(int, pose[12])
               x11, y11, c11 = map(int, pose[11])
               x10, y10, c10 = map(int, pose[10])
               x6, y6, c6 = map(int, pose[8])
               x7, y7, c7 = map(int, pose[7])
               x, y = (x8 + x11) // 2, (y8 + y11) // 2
               zx, zy = (x7 + x6) // 2, (y6 + y7) // 2
               if td == -1:
                   tx, ty, td = x, y, depth[y][x]
                   mx, my, md = x10, y10, depth[y10][x10]
                   ux, uy, ud = zx, zy, depth[zy][zx]
                  
               else:
                   if depth[y][x] > 0 and depth[y][x] < td:
                       tx, ty, td = x, y, depth[y][x]
                       mx, my, md = x10, y10, depth[y10][x10]
                       ux, uy, ud = zx, zy, depth[zy][zx]
                  
           if td!= -1:
               cv2.circle(frame, (tx, ty), 12, (0, 255, 0), -1)
               cv2.circle(frame, (mx, my), 12, (0, 0, 255), -1)
               cv2.circle(frame, (ux, uy), 12, (255, 0, 0), -1)


       boxes = model(frame)[0].boxes
   # 只保留信心度高的框
       valid_boxes = []
       for conf, xyxy in zip(boxes.conf, boxes.xyxy):
           if conf < 0.5: continue
           x1, y1, x2, y2 = map(int, xyxy)
           valid_boxes.append([x1, y1, x2, y2])
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


       # 狀態機邏輯
       print(state)
       if state == State.WAIT_FOR_HUMAN:
           if td != -1:
               rospy.loginfo("👀 偵測到人，進入詢問階段")
               pub_say.publish("Hello what is your name?")
               rospy.sleep(1)
               wait_start_time = rospy.get_time()
               state = State.WAIT_FOR_REPLY


       elif state == State.WAIT_FOR_REPLY:
           if rospy.get_time() - wait_start_time > 6:
               rospy.loginfo("⏰ 超時，重新等待人出現")
               state = State.WAIT_FOR_HUMAN
           elif voice_text != "":
               for name in nameList:
                   if name in voice_text.lower():
                       rospy.sleep(1)
                       pub_say.publish("Nice to meet you, %s ." % name)
                       state = State.TAKE_BAG
                       voice_test =""
       elif state == State.TAKE_BAG:
           if not asked_bag:
               pub_say.publish("Which one is your bag?")
               rospy.sleep(1)
               asked_bag = True
           wait_start_time = rospy.get_time()
           a = ux - mx
           b = my - uy
           print(f"DEBUG: a={a}, b={b}, tx={tx}, mx={mx}, ty={ty}, my={my}")
           if b <  0:
               if a > 0:
                   hund = 'right'
                   pub_say.publish("I will get the bag of right.")
                   rospy.sleep(1)
                   state = State.WAIT_FOR_REPLY1
               else:
                   hund = 'left'
                   pub_say.publish("I will get the bag of left.")
                   rospy.sleep(1)
                   state = State.WAIT_FOR_REPLY1
           # 測試：如果都沒有進入，給一個 fallback
       elif state == State.WAIT_FOR_REPLY1:
           if rospy.get_time() - wait_start_time > 30:
               rospy.loginfo("⏰ 超時，重新等待人出現")
               state = State.TAKE_BAG
               asked_bag = False # 回到 TAKE_BAG 時才會再問 cmd.linear.x = 0






           # elif tx != -1 and ty != -1:
              
               # e_angle = 320 - tx
               # if abs(e_angle) < 30:
               #     e_angle = 0
               # v_angle = 0.002 * e_angle
               # v_angle = min(max(-0.2, v_angle), 0.2)
               # cmd.angular.z = v_angle


               # # 以深度為基準，調整前進
               # desired_dist = 800  # 你想距離物品多近（mm），視你的深度資料調整
               # e_dist = td - desired_dist
               # if abs(e_dist) < 50:
               #     e_dist = 0
               # v_forward = -0.002 * e_dist
               # v_forward = min(max(-0.1, v_forward), 0.15)
               # cmd.linear.x = v_forward


               # # 到達目標
               # if abs(e_dist) < 70 and abs(e_angle) < 50:
               #     pub_say.publish("I have arrived at your bag.")
               #     state = State.WAIT_FOR_FOLLOW
               #     # (或 TASK_DONE，依流程設計)
               #     cmd.linear.x = 0
               #     cmd.angular.z = 0
          
           else:
               # 沒有鎖定，回去等指向
               # state = State.TAKE_BAG
               q2 = td-800
               if abs(q2) < 120:
                   q2 = 0
                   state = State.TURN_TO_THE_BAG
               z2 = 0.005
               v3 = -z2 * q2
               v3 = min(max(-0.1, v3), 0.1)
               cmd.linear.x = v3
          
              
       elif state == State.TURN_TO_THE_BAG:
           # 第一次進入狀態時，記錄開始時間
           if turn_start_time is None:
               turn_start_time = rospy.get_time()
          
           elapsed = rospy.get_time() - turn_start_time


           # 判斷要轉右還是轉左
           if hund == 'right':
               if elapsed < 2.0:
                   cmd.angular.z = 0.2
                  
               else:
                   cmd.angular.z = 0.0    # 停止轉動
                   rospy.loginfo("右轉1秒結束，進下一狀態")
                   state = State.WAIT_FOR_BAG  # 或你的下一個狀態
                   turn_start_time = None         # 重設
           elif hund == 'left':
               if elapsed < 2.0:
                   cmd.angular.z = -0.2   # 左轉（負角速度，依你底盤規則調整）
                  
               else:
                   cmd.angular.z = 0.0    # 停止轉動
                   rospy.loginfo("左轉1秒結束，進下一狀態")
                  
                   state = State.WAIT_FOR_BAG  # 或你的下一個狀態
                   turn_start_time = None         # 重設
              
           else:
               # 沒有指定 left/right 預設不動
               cmd.angular.z = 0.0
       elif  state == State.WAIT_FOR_BAG:
           pub_say.publish("Please put your bag in my body.")    # 右轉（正角速度，依你底盤規則調整）
           rospy.sleep(1)
           state = State.WAIT_FOR_FOLLOW


       elif state == State.WAIT_FOR_FOLLOW:
           if 'follow' in voice_text:
               voice_test =""
               pub_say.publish("I will follow you.")
               state = State.FOLLOW_PERSON




       elif state == State.FOLLOW_PERSON:
           cv2.circle(frame, (tx, ty), 12, (0, 255, 0), -1)
           print("llllllllllllllllllllllllllllllllllllllllllllllllll")
           e = 320 - tx
           if abs(e) < 40:
               e = 0
           p = 0.1
           v =  p * e
           v = min(max(-0.2, v), 0.2)
           cmd.angular.z = v


           e2 = td-800
           if abs(e2) < 50:
               e2 = 0
           p2 = 0.005
           v2 = -p2 * e2
           v2 = min(max(-0.1, v2), 0.1)
           cmd.linear.x = v2


           if "stop" in voice_text:
               cmd.linear.x = 0
               cmd.angular.z = 0
               pub_say.publish("Please take your bag.")
               state = State.call_back_home
               voice_text = ""
       elif state == State.call_back_home:
           move_base.move_to(-1.02,-0.00172,0.00417)
           state = State.back_home
       elif state == State.back_home:
           code = move_base.status_code
           if code == 3:
               state = State.TASK_DONE
       elif state == State.TASK_DONE:
           cmd.linear.x = 0
           cmd.angular.z = 0


       # 發佈控制與介面
       pub_cmd.publish(cmd)
       print(cmd)
       cv2.imshow("FSM Trackin", frame)
       if cv2.waitKey(1) in [27, ord('q')]:
           break


       rate.sleep()


   rospy.loginfo("Node Ended.")
   rospy.loginfo("ros_tutorial node end!")
   cv2.destroyAllWindows()
