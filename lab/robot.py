# robot.py

import numpy as np

class Robot:
    """Represents a general fixed-base kinematic chain."""

    def __init__(self, link_lengths, link_masses, joint_masses, end_effector_mass):
        """
        Constructor: Initializes a new robot with the specified parameters.

        Parameters:
        link_lengths (np.ndarray): 1D array of link lengths.
        link_masses (np.ndarray): 1D array of link masses.
        joint_masses (np.ndarray): 1D array of joint masses.
        end_effector_mass (float): Mass of the end effector.
        """
        # Validate inputs (Do not modify this section)
        if link_lengths.ndim != 1:
            raise ValueError(f'Invalid link_lengths: Should be a 1D array, is {link_lengths.shape}.')
        if link_masses.ndim != 1:
            raise ValueError(f'Invalid link_masses: Should be a 1D array, is {link_masses.shape}.')
        if joint_masses.ndim != 1:
            raise ValueError('Invalid joint_masses: Should be a 1D array.')
        if not isinstance(end_effector_mass, (int, float)):
            raise ValueError('Invalid end_effector_mass: Should be a number.')

        self.dof = link_lengths.shape[0]

        if link_masses.shape[0] != self.dof:
            raise ValueError('Invalid number of link masses: should match number of link lengths.')

        if joint_masses.shape[0] != self.dof:
            raise ValueError('Invalid number of joint masses: should match number of link lengths.')

        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.joint_masses = joint_masses
        self.end_effector_mass = end_effector_mass

    def forward_kinematics(self, thetas):
        """
        Returns the forward kinematic map for each frame, one for the base of
        each link, and one for the end effector. Link i is given by frames[:, :, i],
        and the end effector frame is frames[:, :, -1].

        Parameters:
        thetas (np.ndarray): 1D array of joint angles.

        Returns:
        np.ndarray: Frames, shape (3, 3, dof + 1)
        """
        if thetas.ndim != 1:
            raise ValueError('Expecting a 1D array of joint angles.')

        if thetas.shape[0] != self.dof:
            raise ValueError(f'Invalid number of joints: {thetas.shape[0]} found, expecting {self.dof}')

        # Allocate frames array
        frames = np.zeros((3, 3, self.dof + 1))
        n = self.dof

        # --------------- BEGIN STUDENT SECTION (Implement forward kinematics) ---------------
        # TODO: Implement the forward kinematics to compute the frames for each link and the end effector.

        # Hints:
        # - Use homogeneous transformation matrices.
        # - Initialize the first frame.
        # - Loop through each link to compute the transformations.
        # - Remember that the end effector frame does not include an additional rotation.

        # Your code starts here

        ...

        # Your code ends here
        # --------------- END STUDENT SECTION ------------------------------------------------

        return frames

    def fk(self, thetas):
        """Shorthand for returning the forward kinematics."""
        return self.forward_kinematics(thetas)

    def end_effector(self, thetas):
        """
        Returns [x; y; theta] for the end effector given a set of joint angles.

        Parameters:
        thetas (np.ndarray): 1D array of joint angles.

        Returns:
        np.ndarray: End effector position and orientation.
        """
        # Do not modify this method
        frames = self.fk(thetas)
        H_0_ee = frames[:, :, -1]

        x = H_0_ee[0, 2]
        y = H_0_ee[1, 2]
        th = np.arctan2(H_0_ee[1, 0], H_0_ee[0, 0])

        ee = np.array([x, y, th])
        return ee

    def ee(self, thetas):
        """Shorthand for returning the end effector position and orientation."""
        return self.end_effector(thetas)

    def jacobians(self, thetas):
        """
        Compute the Jacobians for each frame.

        Parameters:
        thetas (np.ndarray): Joint angles, shape (dof,)

        Returns:
        np.ndarray: Jacobians, shape (3, dof, dof+1)
        """
        if thetas.shape != (self.dof,):
            raise ValueError(f'Invalid thetas: Expected shape ({self.dof},), got {thetas.shape}.')

        jacobians = np.zeros((3, self.dof, self.dof + 1))
        epsilon = 0.001

        # --------------- BEGIN STUDENT SECTION (Implement Jacobians) ---------------
        # TODO: Implement the numerical computation of the Jacobians

        # Hints:
        # - Perturb each joint angle by epsilon and compute the forward kinematics.
        # - Compute numerical derivatives for x and y positions.
        # - Determine the rotational component based on joint contributions.

        # Your code starts here

        ...

        # Your code ends here
        # --------------- END STUDENT SECTION ----------------------------------------

        return jacobians

    def inverse_kinematics(self, initial_thetas, goal_position):
        """
        Returns the joint angles which minimize a simple squared-distance cost function.

        Parameters:
        initial_thetas (np.ndarray): Initial joint angles, shape (dof,)
        goal_position (np.ndarray): Goal position, shape (2,) or (3,)

        Returns:
        np.ndarray: Joint angles that achieve the goal position.
        """
        # Make sure that all the parameters are what we're expecting.
        # This helps catch typos and other lovely bugs.
        if initial_thetas.shape != (self.dof,):
            raise ValueError(f'Invalid initial_thetas: Expected shape ({self.dof},), got {initial_thetas.shape}.')

        if goal_position.shape[0] not in [2, 3]:
            raise ValueError(f'Invalid goal_position: Expected length 2 or 3, got {goal_position.shape}.')

        # Allocate a variable for the joint angles during the optimization;
        # begin with the initial condition
        thetas = initial_thetas.copy()

        # Step size for gradient update
        step_size = 0.1

        # Once the norm (magnitude) of the computed gradient is smaller than
        # this value, we stop the optimization
        stopping_condition = 0.00005

        # Also, limit to a maximum number of iterations.
        max_iter = 100
        num_iter = 0

        # --------------- BEGIN STUDENT SECTION ----------------------------------
        # Run gradient descent optimization
        while num_iter < max_iter:
            # Compute the gradient for either an [x, y] goal or an [x, y, theta] goal,
            # using the current value of 'thetas'.
            # TODO: Fill in the gradient of the squared distance cost function
            # HINT: Use the 'self.end_effector' function and the 'self.jacobians' function
            # to help solve this problem

            if goal_position.shape[0] == 2:  # [x, y] goal
                # Your code starts here

                cost_gradient = np.zeros(self.dof)
                # Compute the current end effector position
                # FK = self.end_effector(thetas)
                # Compute the difference between current position and goal
                # d = FK[:2] - goal_position
                # Compute the Jacobian
                # J = self.jacobians(thetas)[:, :, -1]
                # Compute the cost gradient
                # cost_gradient = ...

                ...

                # Your code ends here

            else:  # [x, y, theta] goal
                # Your code starts here

                cost_gradient = np.zeros(self.dof)
                # Compute the current end effector pose
                # FK = self.end_effector(thetas)
                # Compute the difference between current pose and goal
                # d = FK - goal_position
                # Compute the Jacobian
                # J = self.jacobians(thetas)[:, :, -1]
                # Compute the cost gradient
                # cost_gradient = ...

                ...

                # Your code ends here

            # Update 'thetas'
            # Your code starts here

            # thetas = thetas - step_size * cost_gradient

            ...

            # Your code ends here

            # Check stopping condition, and return if it is met.
            # Your code starts here

            # if np.linalg.norm(cost_gradient) < stopping_condition:
            #     return thetas

            ...

            # Your code ends here

            num_iter += 1
        # --------------- END STUDENT SECTION ------------------------------------

        return thetas
