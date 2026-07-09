import roslibpy
import time
client = roslibpy.Ros(host='192.168.167.163', port=9090)
client.run()
print(client.is_connected)

client.run(timeout=5)

print('sent')
roslibpy.Service(client, f"/D02/spot/duck",
                    "std_srvs/Trigger").call(roslibpy.ServiceRequest())

time.sleep(10)

print('sent')
roslibpy.Service(client, f"/D02/spot/duck",
                    "std_srvs/Trigger").call(roslibpy.ServiceRequest())

time.sleep(10)
print('close')
client.terminate()