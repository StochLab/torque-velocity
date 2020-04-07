import numpy as np
from math import sqrt, sin, cos, atan2, radians, copysign
# from future import 
import time
import ik_class
import matplotlib.pyplot as plt
import matplotlib.axes._subplots as axe
import npyscreen
import threading
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from multiprocessing import Process

jump_height = 0.61
extension_height = 0.55
retracted_height = 0.05
g = -9.81
joint_angles = [0, 0]
mass = 12.5
L_0 = 0.3
theta_0 = 45

kv = 90
kt = 60 / (2 * np.pi * kv)

knee_torques = np.empty(shape=1)
hip_torques = np.empty(shape=1)

knee_velocities = np.empty(shape=1)
hip_velocities = np.empty(shape=1)

x_motion = np.empty(shape=1)
y_motion = np.empty(shape=1)

hip_powers = knee_powers = np.empty(shape=1)
hip_currents = knee_currents = np.empty(shape=1)

hip_powers = np.delete(hip_powers, 0, axis=0)
knee_powers = np.delete(knee_powers, 0, axis=0)

knee_torques = np.delete(knee_torques, 0, axis=0)
hip_torques = np.delete(hip_torques, 0, axis=0)
knee_velocities = np.delete(knee_velocities, 0, axis=0)
hip_velocities = np.delete(hip_velocities, 0, axis=0)
x_motion = np.delete(x_motion, 0, axis=0)
y_motion = np.delete(y_motion, 0, axis=0)

knee_positive_points = hip_positive_points = []
stall_torque = np.array([32, 0])
no_load_speed = np.array([0, 28])

max_knee_torque = max_hip_torque = 0
min_knee_torque = min_hip_torque = 10000

max_knee_current = max_hip_current = 0
min_knee_current = min_hip_current = 10000

pos_power_start = 0
valid_workspace = 0
global sol

class form_object(npyscreen.Form):
    def plot(self, sol1):
        global hip_velocities, hip_torques, knee_velocities, knee_torques, knee_tau_vel, hip_tau_vel, xy_motion, valid_workspace, times, knee_powers, hip_powers, knee_currents, hip_currents, knee_positive_points, hip_positive_points

        self.ax[0][0].cla()
        self.ax[1][0].clear()
        self.ax[0][1].clear()
        self.ax[1][1].clear()

        self.ax[0][0].plot(-sol1[:valid_workspace, 0],-sol1[:valid_workspace,1])
        # self.ax[0][0].plot(sol1[pos_power_start:, 0], sol1[pos_power_start:, 1], linewidth=3)
        self.ax[0][0].scatter(-sol1[knee_positive_points, 0], -sol1[knee_positive_points,1], s=30, c='blue', edgecolors='none', label='Positive knee power')
        self.ax[0][0].scatter(-sol1[hip_positive_points, 0], -sol1[hip_positive_points,1], s=15, c='orange', edgecolors='none', label='Positive hip power')
        # self.ax[0][0].scatter(-sol[valid_workspace, 0], -sol[valid_workspace, 1],c='green', edgecolors='none')
        self.ax[0][0].scatter(-sol[max_hip_torque_pos, 0], -sol[max_hip_torque_pos, 1],c='green', edgecolors='none', label='Max hip torque position')
        self.ax[0][0].scatter(-sol[max_knee_torque_pos, 0], -sol[max_knee_torque_pos, 1],c='red', edgecolors='none', label='Max knee torque position')
        self.ax[0][0].legend(fontsize='x-small')
        self.ax[0][0].set_xlim(-1, 1)
        self.ax[0][0].set_ylim(0, 1)
        self.ax[0][0].set(xlabel='x', ylabel='y', title='Trajectory')
        self.ax[0][0].grid()

        self.ax[0][1].plot(no_load_speed, stall_torque)
        self.ax[0][1].plot(hip_velocities, hip_torques)
        self.ax[0][1].set_xlim(left=0)
        self.ax[0][1].set_ylim(bottom=0)
        self.ax[0][1].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)', title='Hip Torque-Velocity')
        self.ax[0][1].grid()

        self.ax[1][0].plot(no_load_speed, stall_torque)
        self.ax[1][0].plot(knee_velocities, knee_torques)
        self.ax[1][0].set_xlim(left=0)
        self.ax[1][0].set_ylim(bottom=0)
        self.ax[1][0].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)', title='Knee Torque-Velocity')
        self.ax[1][0].grid()

        self.ax[1][1].plot(times[:valid_workspace], knee_powers[:valid_workspace], label='Knee motor power')
        self.ax[1][1].plot(times[:valid_workspace], hip_powers[:valid_workspace], label='Hip motor power')
        self.ax[1][1].set(xlabel='Time (s)', ylabel='Motor power (W)', title='Motor Power')
        self.ax[1][1].grid()
        self.ax[1][1].legend(fontsize='x-small', framealpha=0.4)

        plt.subplot(3,1,3)
        plt.cla()
        plt.plot(times[:valid_workspace], knee_currents[:valid_workspace], label='Knee current')
        plt.plot(times[:valid_workspace], hip_currents[:valid_workspace], label='Hip current')
        plt.gca().set(xlabel='Time (s)', ylabel='Motor current (A)')
        plt.legend(fontsize='x-small', framealpha=0.5)
        plt.grid()

        plt.pause(0.001)
    
    def init_animate_all_plots(self):
        self.ax[0][0].cla()
        self.ax[1][0].clear()
        self.ax[0][1].clear()
        # self.ax[1][1].clear()

        self.xy_motion, = self.ax[0][0].plot([], [])
        self.ax[0][0].set_xlim(-1, 1)
        self.ax[0][0].set_ylim(0, 1)
        self.ax[0][0].set(xlabel='x', ylabel='y', title='Trajectory')
        self.ax[0][0].grid()

        self.hip_tau_vel, self.ideal_hip_line = self.ax[0][1].plot([], [], [], [])
        self.ax[0][1].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)', title='Hip Torque-Velocity')
        self.ax[0][1].grid()
        self.ax[0][1].set_xlim(left=0)
        self.ax[0][1].set_ylim(bottom=0)

        self.knee_tau_vel, self.ideal_knee_line= self.ax[1][0].plot([], [], [], [])
        self.ax[1][0].set(xlabel='Velocity (rad/s)', ylabel='Torque (N-m)', title='Knee Torque-Velocity')
        self.ax[1][0].grid()
        self.ax[1][0].set_xlim(left=0)
        self.ax[1][0].set_ylim(bottom=0)

        return self.xy_motion, self.hip_tau_vel, self.knee_tau_vel, self.ideal_hip_line, self.ideal_knee_line,

    def update_animate_all_plots(self, frame):
        global sol
        self.ax[0][0].scatter(-sol[max_hip_torque_pos, 0], -sol[max_hip_torque_pos, 1],c='green')
        self.ax[0][0].scatter(-sol[max_knee_torque_pos, 0], -sol[max_knee_torque_pos, 1],c='red')
        self.ideal_hip_line.set_data(no_load_speed, stall_torque)
        self.ideal_knee_line.set_data(no_load_speed, stall_torque)
        self.xy_motion.set_data(-sol[:frame, 0], -sol[:frame, 1])
        self.hip_tau_vel.set_data(hip_velocities[:frame], hip_torques[:frame])
        self.knee_tau_vel.set_data(knee_velocities[:frame], knee_torques[:frame])

        return self.xy_motion, self.hip_tau_vel, self.knee_tau_vel, self.ideal_hip_line, self.ideal_knee_line,

    def animate_plots(self):
        global valid_workspace
        self.ani = FuncAnimation(self.fig, self.update_animate_all_plots, frames=range(valid_workspace), init_func = self.init_animate_all_plots, blit = False, interval = 0, repeat = False)

        plt.show(block=False)

    # def while_editing(self, theta0_slider):
    #     # global main_thread
    #     # spring_mass_motion()

    #     main_thread = threading.Thread(target = self.animate_plots)

    #     if main_thread.is_alive:
    #         main_thread.run()

    #     else:
    #         main_thread.start()

    def create(self):
        global ux_slider, uy_slider, L0_slider, theta0_slider, k_slider, mass_slider, peak_hip_torque, peak_knee_torque, peak_hip_current, peak_knee_current, peak_xy_position, debug

        ux_slider = self.add(npyscreen.TitleSlider, name = "ux:", value = 0.0, out_of = 5, step = 0.1)
        uy_slider = self.add(npyscreen.TitleSlider, name = "uy:", value = 1.7, out_of = 5, step = 0.1)
        L0_slider = self.add(npyscreen.TitleSlider, name = "L0:", value = 0.3, out_of = 0.5, step = 0.1)
        theta0_slider = self.add(npyscreen.TitleSlider, name = "theta_0:", value = 90, out_of = 90, step = 2)
        k_slider = self.add(npyscreen.TitleSlider, name = "k:", value = 100, out_of = 1000, step = 10)
        mass_slider = self.add(npyscreen.TitleSlider, name = "Mass:", value = 12.5, out_of = 15, step = 1)

        peak_hip_torque = self.add(npyscreen.TitleFixedText, name = "Peak hip torque")
        peak_knee_torque = self.add(npyscreen.TitleFixedText, name = "Peak knee torque")
        peak_hip_current = self.add(npyscreen.TitleFixedText, name = "Peak hip current, power, RMS power")
        peak_knee_current = self.add(npyscreen.TitleFixedText, name = "Peak knee current, power, RMS Power")
        debug = self.add(npyscreen.TitleFixedText, name = "Total Energy")

        # plt.ion()
        self.fig, self.ax = plt.subplots(ncols=2,nrows=3)

    def adjust_widgets(self):
        global sol

        spring_mass_motion()
        peak_hip_torque.display()
        peak_knee_torque.display()
        debug.display()
        peak_hip_current.display()
        peak_knee_current.display()
        self.plot(sol)

    def while_waiting(self):
        global peak_hip_torque, peak_knee_torque
        peak_hip_torque.display()
        peak_knee_torque.display()
        debug.display()

    def afterEditing(self):
        self.parentApp.setNextForm(None)

class App(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN', form_object, name = "RBCCPS MPC CONTROLLER")

def function(a, t, x_0, y_0, timestep):
    global k_slider, mass_slider
    k = k_slider.value
    x, y, u_x, u_y = a

    # print x, y

    k_x = abs(k * cos(atan2(y_0 - y,x_0 - x)))
    k_y = abs(k * sin(atan2(y_0 - y,x_0 - x)))

    dx_dt = u_x
    dy_dt = u_y

    dvx_dt = k_x / mass_slider.value * (x_0 - x)# + u_x
    dvy_dt = k_y / mass_slider.value * (y_0 - y) + g# + u_y

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
    print(takeoff_vel)

    print(a_y)
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
        print(x)
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
        print("Torque:",joint_torques)
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
    global sol, max_hip_torque, max_knee_torque, max_knee_torque_pos, max_hip_torque_pos, peak_hip_torque, peak_knee_torque, pos_power_start, hip_positive_points, knee_positive_points, valid_workspace, debug, times, hip_torques, knee_torques, hip_velocities, knee_velocities, knee_powers, hip_powers, knee_currents, hip_currents, peak_hip_current, peak_knee_current

    debug.value = ""
    max_knee_torque = max_hip_torque = 0
    min_knee_torque = min_hip_torque = 10000

    ik = ik_class.Serial2RKin()

    knee_torques = np.empty(shape=1)
    hip_torques = np.empty(shape=1)

    knee_velocities = np.empty(shape=1)
    hip_velocities = np.empty(shape=1)

    hip_powers = knee_powers = np.empty(shape=1)
    hip_currents = knee_currents = np.empty(shape=1)

    hip_powers = np.delete(hip_powers, 0, axis=0)
    knee_powers = np.delete(knee_powers, 0, axis=0)

    hip_currents = np.delete(hip_currents, 0, axis=0)
    knee_currents = np.delete(knee_currents, 0, axis=0)

    knee_torques = np.delete(knee_torques, 0, axis=0)
    hip_torques = np.delete(hip_torques, 0, axis=0)
    knee_velocities = np.delete(knee_velocities, 0, axis=0)
    hip_velocities = np.delete(hip_velocities, 0, axis=0)

    seconds = 0.5
    steps = 50
    times = np.linspace(0, seconds, steps)
    timestep = seconds/steps

    x0 =  L0_slider.value * cos(radians(-theta0_slider.value))
    y0 =  L0_slider.value * sin(radians(-theta0_slider.value))

    #Initial values:x0, y0, ux0, uy0
    A_0 = [x0, y0, -ux_slider.value, uy_slider.value]
    # Initial length of spring in x and y specified in args
    sol = odeint(function, A_0, times, args=(x0, y0, timestep))
    i, j, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    first_point = True

    count = 0
    knee_positive_points = hip_positive_points = []

    for x, y, v_x, v_y in zip(i, j, vx, vy):
        valid, joint_angles = ik.inverseKinematics(np.array([x,y]), branch=1)

        # if not valid:
        #     valid_workspace = count
        #     break;

        # print(u_y - v_y)

        if y < y0:
            valid_workspace = count
            break;

        jacobian = ik.Jacobian(joint_angles) 

        if(first_point == False):
            a_x = (v_x - u_x) / timestep
            a_y = (v_y - u_y) / timestep

            # print(a_y)
            # if v_y <= 0:
            #     valid_workspace = count
            #     break;

            joint_torques = np.zeros(2)
            joint_torques = jacobian.T.dot(np.array([mass_slider.value * a_x, -mass_slider.value * (a_y-g)]))

            theta_dot = np.linalg.inv(jacobian).dot(np.array([v_x, v_y]))

            if(abs(joint_torques[0]) > max_hip_torque):
                max_hip_torque = joint_torques[0]
                max_hip_torque_pos = count

            if(abs(joint_torques[1]) > max_knee_torque):
                max_knee_torque = joint_torques[1]
                max_knee_torque_pos = count

            if(joint_torques[0] * theta_dot[0] > 0):
                hip_torques = np.append(hip_torques, abs(joint_torques[0]))
                hip_velocities = np.append(hip_velocities, abs(theta_dot[0]))

            else:
                hip_torques = np.append(hip_torques, -1)
                hip_velocities = np.append(hip_velocities, -1)

            if(joint_torques[1] * theta_dot[1] > 0):
                knee_torques = np.append(knee_torques, abs(joint_torques[1]))
                knee_velocities = np.append(knee_velocities, abs(theta_dot[1]))

            else:
                knee_torques = np.append(knee_torques, -1)
                knee_velocities = np.append(knee_velocities, -1)    

            knee_motor_torque = joint_torques[1] / 8
            knee_current = knee_motor_torque / kt

            hip_motor_torque = joint_torques[0] / 8
            hip_current = hip_motor_torque / kt      

            knee_currents = np.append(knee_currents, knee_current)
            hip_currents = np.append(hip_currents, hip_current)

            total_power = theta_dot[1] * joint_torques[1] + theta_dot[0] * joint_torques[0]
            hip_power = theta_dot[0] * joint_torques[0]
            knee_power = theta_dot[1] * joint_torques[1]

            knee_powers = np.append(knee_powers, knee_power)
            hip_powers = np.append(hip_powers, hip_power)
            # print(copysign(1,total_power))

            if(total_power > 0):
                pos_power_start = count

            if(hip_power > 0):
                hip_positive_points.append(count)
                # print(count)

            if(knee_power > 0):
                knee_positive_points.append(count)

            # print(mass_slider.value * 9.81 * (y))

            count = count + 1

        if(first_point == True):    
            first_point = False

        # hip_positive_points = [0,1,2,3,4,8,9,10,11,15,16,17,18]
        u_x = v_x
        u_y = v_y

        debug.value = debug.value + " " + str(mass_slider.value * 9.81 * (y) + 0.5 * k_slider.value * (y0 - y) ** 2 + 0.5 * mass_slider.value * v_y ** 2) 

    knee_power_rms = np.sqrt(np.mean(np.square(knee_powers)))
    hip_power_rms = np.sqrt(np.mean(np.square(hip_powers)))

    peak_hip_current.value = str(max(hip_currents))+" "+str(max(hip_powers))+" "+str(hip_power_rms)
    peak_knee_current.value = str(max(knee_currents))+" "+str(max(abs(knee_powers)))+" "+str(knee_power_rms)

    peak_knee_torque.value = str(max_knee_torque)
    peak_hip_torque.value = str(max_hip_torque)

if __name__ == '__main__':
    # vertical_jump()
    # horizontal_move()
    # sine_velocity() 

    app = App().run()
    # spring_mass_motion()