import numpy as np
from math import sqrt, sin, cos, atan2
import ik_class
import matplotlib.pyplot as plt
import npyscreen
from scipy.integrate import odeint

jump_height = 0.61
extension_height = 0.55
retracted_height = 0.05
g = -9.81
joint_angles = [0, 0]
mass = 12.5
knee_torques = np.empty(shape=1)
hip_torques = np.empty(shape=1)

knee_velocities = np.empty(shape=1)
hip_velocities = np.empty(shape=1)

x_motion = np.empty(shape=1)
y_motion = np.empty(shape=1)

knee_torques = np.delete(knee_torques, 0, axis=0)
hip_torques = np.delete(hip_torques, 0, axis=0)
knee_velocities = np.delete(knee_velocities, 0, axis=0)
hip_velocities = np.delete(hip_velocities, 0, axis=0)
x_motion = np.delete(x_motion, 0, axis=0)
y_motion = np.delete(y_motion, 0, axis=0)


stall_torque = np.array([32, 0])
no_load_speed = np.array([0, 28])

class form_object(npyscreen.Form):
    def create(self):
        global speed_slider, nsteps_slider, interval_slider, debug, debug2
        main_thread.start()

        nsteps_slider = self.add(npyscreen.TitleSlider, name = "No. of steps:", value = 15, out_of = 50)
        interval_slider = self.add(npyscreen.TitleSlider, name = "Time interval:", value = .2, out_of = 10, step = 0.1)
        speed_slider = self.add(npyscreen.TitleSlider, name = "Speed:", value = 0.5, out_of = 3, step = 0.1)
        debug = self.add(npyscreen.TitleText, name="Average time:")
        debug2 = self.add(npyscreen.TitleText, name="Instantaneous time:")

    def afterEditing(self):
        global main_thread
        self.parentApp.setNextForm(None)
        main_thread.do_run = False
        main_thread.join()

class App(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN', form_object, name = "RBCCPS MPC CONTROLLER")

def function(a, t, x_0, y_0, timestep):
	k = 600
	x, y, u_x, u_y = a

	# print x, y

	k_x = abs(k * cos(atan2(y_0 - y,x_0 - x)))
	k_y = abs(k * sin(atan2(y_0 - y,x_0 - x)))

	dx_dt = u_x
	dy_dt = u_y

	dvx_dt = k_x / mass * (x_0 - x) + u_x
	dvy_dt = k_y / mass * (y_0 - y) + g + u_y

	# print dvy_dt
	return [dx_dt, dy_dt, dvx_dt, dvy_dt]

def vertical_jump():
	#Calculate vel. required for takeoff at extension height
	#v_y^2 = u^2 - 2gS (u = takeoff_vel, v_y = 0)
	S_x = jump_height - extension_height
	takeoff_vel = sqrt(2 * -g * S_x)

	S_x = extension_height - retracted_height
	#Acc. required for reaching takeoff_vel at extension_height
	#v_y^2 - u^2 = 2aS (u = 0, v_y = takeoff_vel)
	a_y = takeoff_vel**2 / (2 * S_x)

	#Time required for full extension
	#v_y = u + at (v_y = takeoff_vel, u = 0)
	t = takeoff_vel/a_y
	print takeoff_vel

	print a_y
	calc_vel_torq_curve(t, 0, 0, a_y, retracted_height, 0)

def horizontal_move():
	v = 1.5
	S_x = 0.24
	t = S_x/v
	height = .5
	calc_vel_torq_curve(t, 0, v, 0, height, S_x)

def sine_velocity():
	# sin_points = sin_points * S_x/6.28318
	global knee_torques, hip_torques, knee_velocities, hip_velocities

	ik = ik_class.Serial2RKin()

	S_x = 0.4 
	sin_points, timestep = np.linspace(0, 2 * 3.14159, num=100, retstep=True)
	avg_vel_y = 0.5
	t = (S_x / avg_vel_y)
	times = np.linspace(0, t, num=100)	
	sin_vel = avg_vel_y + np.sin(2*3.14159/t*times) * 0.5

	# print timestep
	# print sin_points
	# print sin_vel

	prev_x = sin_points[0]
	prev_vel = sin_vel[0]

	prev_vel_x = 0
	prev_vel_y = 0

	a_x = a_y = 0

	for t_inst, vel in zip(times, sin_vel):
		v_x = vel
		v_y = 0

		x = -(S_x/2) + t_inst * v_x
		print x
		a_x = (v_x - prev_vel_x)/t_inst
		a_y = (v_y - prev_vel_y)/t_inst

		valid, joint_angles = ik.inverseKinematics(np.array([x,0.5]), branch=1)

		joint_torques = np.zeros(2)
		jacobian = ik.Jacobian(joint_angles) 
		joint_torques =  jacobian.T.dot(np.array([mass * a_x, -mass * (a_y-g)]))

		#Calculate vel every time instant
		# v_y = a_y * t_inst
		theta_dot = np.linalg.inv(jacobian).dot(np.array([v_x, v_y]))
		# print "Torque:",joint_torques
		knee_torques = np.append(knee_torques, joint_torques[1])
		hip_torques = np.append(hip_torques, joint_torques[0])

		# print "Velocity:",theta_dot
		knee_velocities = np.append(knee_velocities, theta_dot[1])
		hip_velocities = np.append(hip_velocities, theta_dot[0])

		prev_vel_x = v_x
		prev_vel_y = v_y

	# print knee_velocities
	# print knee_torques

	fig, ax = plt.subplots(nrows=2 ,ncols=2)
	# ax[0][0].plot(no_load_speed, stall_torque)
	ax[0][0].plot(times, knee_velocities)
	# ax[0].plot(sin_vel, times)

	ax[0][0].set(xlabel='Time (s)', ylabel='Velocity (rad/s)',
       title='Knee Velocity')
	ax[0][0].grid()

	# ax[0][1].plot(no_load_speed, stall_torque)
	ax[0][1].plot(times, knee_torques)

	ax[0][1].set(xlabel='Time (s)', ylabel='Torque (N-m)',
       title='Knee Torque')
	ax[0][1].grid()

	ax[1][0].plot(no_load_speed, stall_torque)
	ax[1][0].plot(knee_velocities, knee_torques)
	# ax[0].plot(sin_vel, times)

	ax[1][0].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)',
       title='Knee Torque-Velocity')
	ax[1][0].grid()

	ax[1][1].plot(no_load_speed, stall_torque)
	ax[1][1].plot(hip_velocities, hip_torques)

	ax[1][1].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)',
       title='Hip Torque-Velocity')
	ax[1][1].grid()
	plt.show()

def calc_vel_torq_curve(t, a_x, v_x, a_y, retracted_height, horizontal_dist):
	global knee_torques, hip_torques, knee_velocities, hip_velocities
	times = np.linspace(0, t, num=100)	
	ik = ik_class.Serial2RKin()

	for t_inst in times:
		#Calculate body pos. every time instant
		x = -(horizontal_dist/2) + t_inst * v_x
		y = retracted_height + 0.5 * a_y * t_inst ** 2
		valid, joint_angles = ik.inverseKinematics(np.array([x,y]), branch=1)
		# print valid

		joint_torques = np.zeros(2)
		jacobian = ik.Jacobian(joint_angles) 
		joint_torques =  jacobian.T.dot(np.array([0, -mass * (a_y-g)]))

		#Calculate vel every time instant
		v_y = a_y * t_inst
		theta_dot = np.linalg.inv(jacobian).dot(np.array([v_x, v_y]))
		print "Torque:",joint_torques
		knee_torques = np.append(knee_torques, joint_torques[1])
		hip_torques = np.append(hip_torques, joint_torques[0])

		# print "Velocity:",theta_dot
		knee_velocities = np.append(knee_velocities, theta_dot[1])
		hip_velocities = np.append(hip_velocities, theta_dot[0])
		# print joint_angles * 180/3.14159

	# print knee_velocities
	# print knee_torques
	fig, ax = plt.subplots(ncols=2)
	ax[0].plot(no_load_speed, stall_torque)
	ax[0].plot(-knee_velocities, knee_torques)

	ax[0].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)',
       title='Knee Torque-Velocity')
	ax[0].grid()

	ax[1].plot(no_load_speed, stall_torque)
	ax[1].plot(hip_velocities, hip_torques)

	ax[1].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)',
       title='Hip Torque-Velocity')
	ax[1].grid()
	plt.show()

def spring_mass_motion():
	global knee_torques, hip_torques, knee_velocities, hip_velocities, x_motion, y_motion

	ik = ik_class.Serial2RKin()

	#Initial velocity
	# u_x = 0.15
	# u_y = -0.15

	#Initial position
	# x = -0.3
	# y = 0.3

	#Timestep
	# timestep = 0.05

	#Spring constant
	# k = 548
	# time = 0.


	seconds = 1.
	steps = 100.
	times = np.linspace(0, seconds, steps)
	timestep = seconds/steps

	#Initial values:x0, y0, ux0, uy0
	A_0 = [0.3, -0.3, -0.3, 0.3]
	# Initial length of spring in x and y specified in args
	sol = odeint(function, A_0, times, args=(0., -0.3, timestep))

	# while(time < 2):
	# 	#Calculate k in x, y axis
	# 	k_x = k * cos(atan2(y,x))
	# 	k_y = k * sin(atan2(y,x))
	# 	print "k_x=", k_x
	# 	print "k_y=", k_y

	# 	#Calculate acc. for x, y axis
	# 	a_x =  k_x * x / mass
	# 	a_y =  k_y * y / mass + g

	# 	#Calculate final vel. considering spring resistance
	# 	v_x = u_x + a_x * timestep
	# 	v_y = u_y + a_y * timestep

	# 	#Update position
	# 	x = x + v_x * timestep + 0.5 * a_x * timestep ** 2
	# 	y = y + v_y * timestep + 0.5 * a_y * timestep ** 2

	# 	x_motion = np.append(x_motion, x)
	# 	y_motion = np.append(y_motion, y)

	# 	valid, joint_angles = ik.inverseKinematics(np.array([x,y]), branch=1)
	# 	print x, y
	# 	print valid

	# 	if not valid:
	# 		break;

	# 	joint_torques = np.zeros(2)
	# 	jacobian = ik.Jacobian(joint_angles) 
	# 	joint_torques = jacobian.T.dot(np.array([0, -mass * (a_y-g)]))

	# 	theta_dot = np.linalg.inv(jacobian).dot(np.array([v_x, v_y]))
	# 	# print "Torque:",joint_torques
	# 	knee_torques = np.append(knee_torques, joint_torques[1])
	# 	hip_torques = np.append(hip_torques, joint_torques[0])

	# 	# print "Velocity:",theta_dot
	# 	knee_velocities = np.append(knee_velocities, theta_dot[1])
	# 	hip_velocities = np.append(hip_velocities, theta_dot[0])

	# 	u_x = v_x
	# 	u_y = v_y

	# 	time = time + timestep
	print sol
	fig, ax = plt.subplots(ncols=1)

	ax.plot(sol[:, 0], sol[:, 1])
	# ax.plot(x_motion, y_motion)

	ax.set(xlabel='x', ylabel='y',
       title='Trajectory')
	ax.grid()

	# ax[1].plot(hip_velocities, hip_torques)

	# ax[1].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)',
 #       title='Hip Torque-Velocity')
	# ax[1].grid()
	plt.show()


if __name__ == '__main__':
	# vertical_jump()
	# horizontal_move()
	# sine_velocity() 
	spring_mass_motion()