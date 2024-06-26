import time
from algo.Calgo import MazeSolver 
from flask import Flask, request, jsonify
from flask_cors import CORS
# from model import *
from models import objectdetection_yolov8 
import os

from helper import command_generator

app = Flask(__name__)
CORS(app)
# model = load_model()

@app.route('/status', methods=['GET'])
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return jsonify({"result": "ok"})


@app.route('/path', methods=['POST'])
def path_finding():
    """
    This is the main endpoint for the path finding algorithm
    :return: a json object with a key "data" and value a dictionary with keys "distance", "path", and "commands"
    """
    # Get the json data from the request
    content = request.json

    # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction from the json data
    obstacles = content['obstacles']
    big_turn = int(content['big_turn'])
    retrying = content['retrying']
    robot_x, robot_y = content['robot_x'], content['robot_y']
    robot_direction = int(content['robot_dir'])

    # Initialize MazeSolver object with robot size of 20x20, bottom left corner of robot at (1,1), facing north, and whether to use a big turn or not.
    # maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=None)
    maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=big_turn)

    # Add each obstacle into the MazeSolver. Each obstacle is defined by its x,y positions, its direction, and its id
    for ob in obstacles:
        maze_solver.add_obstacle(int(ob['x']), int(ob['y']), int(ob['d']), int(ob['id']))

    start = time.time()
    # Get shortest path
    optimal_order_dp = maze_solver.get_optimal_order_dp(retrying=retrying)
    optimal_path, distance = optimal_order_dp[0], optimal_order_dp[1]
    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")
    
    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)

    # Get the starting location and add it to path_results
    path_results = [optimal_path[0].get_dict()]
    # Process each command individually and append the location the robot should be after executing that command to path_results
    i = 0
    for command in commands:
        if command.startswith("SNAP"):
            continue
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())
    return jsonify({
        "data": {
            'distance': distance,
            'path': path_results,
            'commands': commands
        },
        "error": None
    })


@app.route('/image', methods=['POST'])
def image_predict():
    # save the image file to the uploads folder
    file = request.files['file']
    filename = file.filename
    print('testtt', filename)
    # file.save(os.path.join('uploads', filename))
    # perform image recognition
    # filename format: "<timestamp>_<obstacle_id>.jpeg"
    obstacle_id = file.filename.split("_")[1].strip(".jpg")
    image_id = objectdetection_yolov8.detect(str(file.filename))
    if len(image_id) != 0:
        image_id = image_id[0]
    #image_id = predict_image(filename, model)
    print("Prediction finish")

    result = {
        "obstacle_id": obstacle_id,
        "image_id": image_id
    }

    # only include the "stop" field if the request is for the "navigate around obstacle" feature
    if obstacle_id in ['N', 'S', 'E', 'W']:
        # set stop to True if non-bullseye detected
        result['stop'] = image_id != "10"

    return jsonify(result)

@app.route('/stitch', methods=['GET'])
def stitch_():
    """
    This is the main endpoint for the stitching command. Stitches the images using two different functions, in effect creating two stitches, just for redundancy purposes
    """
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()
    return jsonify({"result": "ok"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
