import pyfirmata

def singleton(cls):
	instances = {}
	def getinstance():
		if cls not in instances:
			instances[cls] = cls()
		return instances[cls]
	return getinstance

@singleton
class PortOpen: 
    def __init__(self):
        self.board = pyfirmata.Arduino('com5')
        self.servo_pinX = self.board.get_pin('d:9:s') 
    def move_servo(self,angle):
        servo_pinX = self.servo_pinX
        servo_pinX.write(angle)


