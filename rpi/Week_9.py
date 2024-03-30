#!/usr/bin/env python3
import json
import queue
import time
from multiprocessing import Process, Manager
from typing import Optional
import os
import requests
from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from consts import SYMBOL_MAP
from logger import prepare_logger
from settings import API_IP, API_PORT
from models import objectdetection_yolov8 
import cv2
import numpy as np


import picamera2

class PiAction:
    """
    Class that represents an action that the RPi needs to take.    
    """

    def __init__(self, cat, value):
        """
        :param cat: The category of the action. Can be 'info', 'mode', 'path', 'snap', 'obstacle', 'location', 'failed', 'success'
        :param value: The value of the action. Can be a string, a list of coordinates, or a list of obstacles.
        """
        self._cat = cat
        self._value = value

    @property
    def cat(self):
        return self._cat

    @property
    def value(self):
        return self._value


class RaspberryPi:
    """
    Class that represents the Raspberry Pi.
    """

    def __init__(self):
        """
        Initializes the Raspberry Pi.
        """
        self.logger = prepare_logger()
        self.android_link = AndroidLink()
        self.stm_link = STMLink()

        self.manager = Manager()

        self.android_dropped = self.manager.Event()
        self.unpause = self.manager.Event()

        self.movement_lock = self.manager.Lock()

        self.android_queue = self.manager.Queue()  # Messages to send to Android
        # Messages that need to be processed by RPi
        self.rpi_action_queue = self.manager.Queue()
        # Messages that need to be processed by STM32, as well as snap commands
        self.command_queue = self.manager.Queue()
        # X,Y,D coordinates of the robot after execution of a command
        self.path_queue = self.manager.Queue()

        self.proc_recv_android = None
        self.proc_recv_stm32 = None
        self.proc_android_sender = None
        self.proc_command_follower = None
        self.proc_rpi_action = None
        self.rs_flag = False
        self.success_obstacles = self.manager.list()
        self.failed_obstacles = self.manager.list()
        self.obstacles_id = 1
        self.current_location = self.manager.dict()
        self.failed_attempt = False
        self.obs = {}

    def start(self):
        """Starts the RPi orchestrator"""
        try:
            ### Start up initialization ###

            self.android_link.connect()
            self.android_queue.put(AndroidMessage(
                'info', 'You are connected to the RPi!'))
            self.stm_link.connect()

            # Define child processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_recv_stm32 = Process(target=self.recv_stm)
            self.proc_android_sender = Process(target=self.android_sender)
            self.proc_command_follower = Process(target=self.command_follower)
            self.proc_rpi_action = Process(target=self.rpi_action)

            # Start child processes
            self.proc_recv_android.start()
            self.proc_recv_stm32.start()
            self.proc_android_sender.start()
            self.proc_command_follower.start()
            self.proc_rpi_action.start()

            self.logger.info("Child Processes started")

            ### Start up complete ###

            # Send success message to Android
            self.android_queue.put(AndroidMessage('info', 'Robot is ready!'))
            self.android_queue.put(AndroidMessage('mode', 'path'))
            self.reconnect_android()
            

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops all processes on the RPi and disconnects gracefully with Android and STM32"""
        self.android_link.disconnect()
        self.stm_link.disconnect()
        self.logger.info("Program exited!")

    def reconnect_android(self):
        """Handles the reconnection to Android in the event of a lost connection."""
        self.logger.info("Reconnection handler is watching...")

        while True:
            # Wait for android connection to drop
            self.android_dropped.wait()

            self.logger.error("Android link is down!")

            # Kill child processes
            self.logger.debug("Killing android child processes")
            self.proc_android_sender.kill()
            self.proc_recv_android.kill()

            # Wait for the child processes to finish
            self.proc_android_sender.join()
            self.proc_recv_android.join()
            assert self.proc_android_sender.is_alive() is False
            assert self.proc_recv_android.is_alive() is False
            self.logger.debug("Android child processes killed")

            # Clean up old sockets
            self.android_link.disconnect()

            # Reconnect
            self.android_link.connect()

            # Recreate Android processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_android_sender = Process(target=self.android_sender)

            # Start previously killed processes
            self.proc_recv_android.start()
            self.proc_android_sender.start()

            self.logger.info("Android child processes restarted")
            self.android_queue.put(AndroidMessage(
                "info", "You are reconnected!"))
            self.android_queue.put(AndroidMessage('mode', 'path'))

            self.android_dropped.clear()

    def recv_android(self) -> None:
        """
        [Child Process] Processes the messages received from Android
        """                        
        while True:
            msg_str: Optional[str] = None
            try:
                msg_str = self.android_link.recv()
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android connection dropped")

            if msg_str is None:
                continue
            print(msg_str)
            message: dict = json.loads(msg_str)
                
            ## Command: Start Moving ##
            if message['cat'] == "manual":
                if message['value'] == "WN01":

                    # Commencing path following                   
                    # if not self.command_queue.empty():
                    self.logger.info("Gyro reset!")
                    self.stm_link.send("RS00\n")
                    # Main trigger to start movement #
                    self.command_queue.put("FC01")
                    self.command_queue.put("SNAP01")
                    self.unpause.set()
                    self.logger.info("Start command received, starting robot on path!")
                    self.android_queue.put(AndroidMessage('info', 'Starting Task 2!'))
                    self.android_queue.put(AndroidMessage('status', 'running'))
                
                elif message['value'] == "WN02":

                    # Commencing path following                   
                    # if not self.command_queue.empty():
                    self.logger.info("Gyro reset!")
                    self.stm_link.send("RS00\n")
                    # Main trigger to start movement #
                    self.command_queue.put("FC01")
                    self.command_queue.put("SNAP21")
                    self.unpause.set()
                    self.logger.info("Start command received, starting robot on path!")
                    self.android_queue.put(AndroidMessage('info', 'Starting Task 2!'))
                    self.android_queue.put(AndroidMessage('status', 'running'))

    def recv_stm(self) -> None:
        """
        [Child Process] Receive acknowledgement messages from STM32, and release the movement lock
        """
        while True:

            print("waiting message")
            message: str = self.stm_link.recv()
            print("msg rec")
            
            if message.startswith("ACK"):
                if self.rs_flag == False:
                    self.rs_flag = True
                    # self.logger.debug("ACK for RS00 from STM32 received.")
                    # continue
                try:
                    self.movement_lock.release()
                    print("move lock released (rec stm)")
                    try:
                        self.retrylock.release()
                        # print("retry lock released")
                    except:
                        pass
                    # self.logger.debug("ACK from STM32 received, movement lock released.")

                    cur_location = self.path_queue.get_nowait()

                    self.current_location['x'] = cur_location['x']
                    self.current_location['y'] = cur_location['y']
                    self.current_location['d'] = cur_location['d']
                    self.logger.info(
                        f"self.current_location = {self.current_location}")
                    self.android_queue.put(AndroidMessage('location', {
                        "x": cur_location['x'],
                        "y": cur_location['y'],
                        "d": cur_location['d'],
                    }))

                except Exception:
                    self.logger.warning("Tried to release a released lock!")
            else:
                self.logger.warning(
                    f"Ignored unknown message from STM: {message}")

    def android_sender(self) -> None:
        """
        [Child process] Responsible for retrieving messages from android_queue and sending them over the Android link. 
        """
        while True:
            # Retrieve from queue
            try:
                message: AndroidMessage = self.android_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.android_link.send(message)
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android dropped")

    def command_follower(self) -> None:
        """
        [Child Process] 
        """
        while True:
            
            self.logger.debug("wait for unpause (command_follower)")

            # Wait for unpause event to be true [Main Trigger]
            try:
                self.logger.debug("wait for retrylock (command_follower)")
                self.retrylock.acquire()
                self.retrylock.release()
                self.logger.debug("retrylock release (command_follower_queue)")

            except:
                self.logger.debug("wait for unpause. cannot get retrylock")
                self.unpause.wait()
            
            # Retrieve next movement command    
            command: str = self.command_queue.get()
            print("COMMAND GOT: ", command)

            
            self.logger.debug("wait for movelock (command follower)")
            # Acquire lock first (needed for both moving, and snapping pictures)
            self.movement_lock.acquire()
            
            # STM32 Commands - Send straight to STM32
            # stm32_prefixes = ("FS", "BS", "FW", "BW", "FL", "FR", "BL",
            #                 "BR", "TL", "TR", "A", "C", "DT", "STOP", "ZZ", "RS")
            stm32_prefixes = ("FC", "FW", "BW")

            if command.startswith(stm32_prefixes):
                self.stm_link.send(command+"\n")
                print("movement lock releasedd")

            # Snap command
            elif command.startswith("SNAP01"):
                self.rpi_action_queue.put(PiAction(cat="snap01", value=None))
                
            elif command.startswith("SNAP21"):
                self.rpi_action_queue.put(PiAction(cat="snap21", value=None))

              
            # End of path
            elif command == "FIN":
                self.stitch()
                self.unpause.clear()
                self.movement_lock.release()
                self.logger.info("Commands queue finished.")
                self.android_queue.put(AndroidMessage("info", "Commands queue finished."))
                self.android_queue.put(AndroidMessage("status", "finished"))
                
            else:
                raise Exception(f"Unknown command: {command}")
    
    def rpi_action(self):
        """
        [Child Process] 
        """
        count = 0
        retry = 0
        while True:
            action: PiAction = self.rpi_action_queue.get()
            self.logger.debug(
                f"PiAction retrieved from queue: {action.cat} {action.value}")
    
            if action.cat == "snap01":
                print("snapping 01")
                res = self.image_predict()
                print(res)
                
                if res['image_id'] == "38": # right arrow
                    if count == 1:
                        self.command_queue.put("FC16")
                        self.command_queue.put("FC19")
                        self.command_queue.put("FIN")
                        self.movement_lock.release()
                        break
                    
                    self.command_queue.put("FC12")
                    self.command_queue.put("SNAP01")
                    count += 1

                    
                elif res['image_id'] == "39": # left_arrow
                    if count == 1:
                        self.command_queue.put("FC15")
                        self.command_queue.put("FC18")
                        self.command_queue.put("FIN")
                        self.movement_lock.release()
                        break

                    self.command_queue.put("FC11")
                    self.command_queue.put("SNAP01")
                    count += 1

                    
                else:
                    self.logger.debug("No detections")
                    if retry == 0:

                        self.command_queue.put("BW05")
                        self.command_queue.put("SNAP01")
                        self.command_queue.put("FW05")
                        retry+=1
                    
                    else:
                        if count == 0:
                            retry = 0
                            self.command_queue.put("FC11")
                            self.command_queue.put("SNAP01")
                            count+=1

                        # guess left arrow
                        
                        else:
                            self.command_queue.put("FC15")
                            self.command_queue.put("FC18")
                            self.command_queue.put("FIN")
                            self.movement_lock.release()
                            break

                        
                        
                self.movement_lock.release()
                try:
                    self.retrylock.release()
                except:
                    pass
                print("movement lock release")
                
            elif action.cat == "snap21":
                print("snapping")
                res = self.image_predict()
                print(res)
                
                if res['image_id'] == "38": # right arrow
                    if count == 1:
                        self.command_queue.put("FC26")
                        self.command_queue.put("FC29")
                        self.command_queue.put("FIN")
                        self.movement_lock.release()
                        break
                    
                    self.command_queue.put("FC22")
                    self.command_queue.put("SNAP21")
                    count += 1

                    
                elif res['image_id'] == "39": # left_arrow
                    if count == 1:
                        self.command_queue.put("FC25")
                        self.command_queue.put("FC28")
                        self.command_queue.put("FIN")
                        self.movement_lock.release()
                        break

                    self.command_queue.put("FC21")
                    self.command_queue.put("SNAP21")
                    count += 1
                             
                else:
                    self.logger.debug("No detections")
                    if retry == 0:

                        self.command_queue.put("BW05")
                        self.command_queue.put("SNAP21")
                        self.command_queue.put("FW05")
                        retry+=1
                    
                    else:
                        # guess left arrow
                        if count == 0:
                            retry = 0
                            self.command_queue.put("FC21")
                            self.command_queue.put("SNAP21")
                            count+=1
                        else:
                            self.command_queue.put("FC25")
                            self.command_queue.put("FC28")
                            self.command_queue.put("FIN")
                            self.movement_lock.release()
                            break
                               
                self.movement_lock.release()
                try:
                    self.retrylock.release()
                except:
                    pass
                print("movement lock release")

                    
                #self.snap_and_rec(obstacle_id_with_signal=action.value)
                
            elif action.cat == "stitch":
                self.request_stitch()

    def clear_queues(self):
        """Clear both command and path queues"""
        while not self.command_queue.empty():
            self.command_queue.get()
        while not self.path_queue.empty():
            self.path_queue.get()
        
    def image_predict(self):
        # save the image file to the uploads folder
        
        print("capturing image...")
        filename = f"{int(time.time())}_{self.obstacles_id}.jpg"
        captured_dir = r"/home/pi/MDP2/src/images/task2/captured"
        
        with picamera2.Picamera2() as camera:
            config = camera.create_preview_configuration(main={"size":(1640,1232)})
            #config = camera.create_preview_configuration(main={"size":(1280,720)})
            camera.configure(config)
            camera.start()
            time.sleep(1)
            img_full_path = os.path.join(captured_dir, filename)
            camera.capture_file(img_full_path)   
            print("image captured")

        print('testtt', filename)
        # perform image recognition
        # filename format: "<timestamp>_<obstacle_id>.jpeg"
        image_id = objectdetection_yolov8.detect(img_full_path, task=2)
       
        result = {
            "obstacle_id": self.obstacles_id,
            "image_id": image_id
        }
        self.obstacles_id += self.obstacles_id

        return result

    def stitch(self):
        print("stitching...")
        file_path = r"/home/pi/MDP2/src/images/task2/results"
        save_dir = r"/home/pi/MDP2/src/images/task2/runs"
        count = 0

        try:
            img_size = (640,640)
        
            captured_img = []
            for file in os.listdir(file_path):
                print(file)
                try:
                    img = cv2.imread(os.path.join(file_path, file))
                except:
                    print("fail to read: ")
                    img = np.empty((640, 640, 3))
                print("img")
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                captured_img.append(img)

            while len(captured_img) %3 != 0:
                captured_img.append(np.zeros((640, 640, 3)))

            grouped = list(zip(*[iter(captured_img)] * 3))

            final_stitched = []
            for i in grouped:
                row = i[0]
                for x in range(len(i)):
                    if x == 0:
                        continue
                    row = np.append(row, i[x], axis=1)
                final_stitched.append(row)

            res = np.vstack(final_stitched)
            cv2.imwrite(os.path.join(save_dir, "stitch.png"), res)
            print("stitched")
        except:
            print("stitching failed")


if __name__ == "__main__":
    rpi = RaspberryPi()
    rpi.start()
