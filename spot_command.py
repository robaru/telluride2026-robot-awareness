#!/usr/bin/env python3
"""Send a movement command to Spot over rosbridge.

Instructions:
1. connect to BrainAirWaves wifi
2. test connection on terminal with "ping 192.168.167.163"
    - something should reply in 50 ms circa.
3. pip install roslibpy

Usage:
    python spot_command.py forward
    python spot_command.py backward
    python spot_command.py left
    python spot_command.py right
    python spot_command.py stop
"""

import argparse
import sys

import roslibpy

SPOT_HOST = "192.168.167.163"
SPOT_PORT = 9090

COMMANDS = {
    "stop": 0,
    "backward": 1,
    "forward": 2,
    "left": 3,
    "right": 4,
}


def send_command(command: str, host: str = SPOT_HOST, port: int = SPOT_PORT) -> None:
    client = roslibpy.Ros(host=host, port=port)
    client.run()

    if not client.is_connected:
        sys.exit(f"Failed to connect to Spot at {host}:{port}")

    topic = roslibpy.Topic(client, "/spot_command/digit", "std_msgs/Int8")
    topic.publish(roslibpy.Message({"data": COMMANDS[command]}))
    print(f"Sent '{command}' ({COMMANDS[command]}) to /spot_command/digit")

    topic.unadvertise()
    client.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a movement command to Spot.")
    parser.add_argument("command", choices=COMMANDS.keys(), help="Command to send")
    parser.add_argument("--host", default=SPOT_HOST, help="Spot ROS host IP")
    parser.add_argument("--port", type=int, default=SPOT_PORT, help="rosbridge port")
    args = parser.parse_args()

    send_command(args.command, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
