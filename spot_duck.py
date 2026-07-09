import roslibpy

client = roslibpy.Ros(host='192.168.167.163', port=9090)
client.run()
print(client.is_connected)

# Force-steal the lease from the teleop controller
take = roslibpy.Service(client, '/D02/spot/take', 'std_srvs/Trigger')
result = take.call(roslibpy.ServiceRequest())
print('take:', result)

# Duck
duck = roslibpy.Service(client, '/D02/spot/duck', 'std_srvs/Trigger')
result = duck.call(roslibpy.ServiceRequest())
print('duck:', result)

# Give the lease back to teleop
release = roslibpy.Service(client, '/D02/spot/release', 'std_srvs/Trigger')
release.call(roslibpy.ServiceRequest())