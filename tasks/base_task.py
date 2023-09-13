
from environment.parameters import *

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim
from omni.isaac.franka import Franka
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import set_stage_units, set_stage_up_axis, is_stage_loading
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts.utils import setStaticCollider
from omni.kit.material.library import get_material_prim_path
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import physicsUtils

from scipy.spatial.transform import Rotation as R
import pxr
from pxr import UsdPhysics, Gf, PhysxSchema, UsdShade, UsdGeom
import time
from omni.isaac.synthetic_utils import SyntheticDataHelper
from abc import ABC
from omni.isaac.franka.controllers import RMPFlowController

class BaseTask(ABC):
    material_library = {}
    viewport_handles = []
    
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        self.cfg = cfg
        self.num_stages = num_stages
        self.horizon = horizon
        self.stage_properties: StageProperties = stage_properties
        self.timeline = omni.timeline.get_timeline_interface()
        self.kit = omni.kit.app.get_app()

        self.cube_camera_is_set = False

        self.objects_list = []
    
    def success(self):
        if hasattr(self, "checker") and self.checker and self.checker.success:
            return True
        
        return False

    def set_up_task(self):
        raise NotImplementedError
    
    def remove_objects(self):
        for prim in self.objects_list:
            delete_prim(prim.GetPath().pathString)
        if is_prim_path_valid('/World'):
            delete_prim('/World')
        if is_prim_path_valid('/Looks'):
            delete_prim('/Looks')
        if is_prim_path_valid('/lula'):
            delete_prim('/lula')
        self.objects_list = []
        self._wait_for_loading()

    def stop(self):
        self.timeline.stop()
        self._wait_for_loading()
        self.remove_objects()
    
    def reset(self,
              robot_parameters = None,
              scene_parameters = None,
              sensor_resolution = (128, 128),
              sensor_types = ["rgb", "depthLinear", "camera", "semanticSegmentation"],
        ):

        self.timeline.stop()
        self.checker = None
        self.kit.update()

        self.stage = omni.usd.get_context().get_stage()
        self.sensor_resolution = sensor_resolution
        self.sensor_types = sensor_types
        
        simulation_context = SimulationContext.instance()

        if robot_parameters is not None:
            self.robot_parameters = robot_parameters

        if scene_parameters is not None:
            self.scene_parameters = scene_parameters
            self.num_envs = len(scene_parameters)

        self.clear()
        self._wait_for_loading()
        self._define_stage_properties()

        if self.use_gpu_physics:
            simulation_context.get_physics_context().enable_gpu_dynamics(self.use_gpu_physics)
            simulation_context.get_physics_context().set_broadphase_type("GPU")
        
        self._load_scene()
        self.robot = self._load_robot()
        self.set_up_task()
        self._wait_for_loading()
        
        self.timeline.play()

        self.kit.update()

        def initialize(robot):
            robot.initialize()
            robot.set_joint_positions(robot._articulation_view._default_joints_state.positions)
            robot.set_joint_velocities(robot._articulation_view._default_joints_state.velocities)
            robot.set_joint_efforts(robot._articulation_view._default_joints_state.efforts)
            add_update_semantics(get_prim_at_path(robot.prim_path), "Robot")
           
            # this is important
            robot.disable_gravity()
            self.kit.update()

        initialize(self.robot)
 
        self.articulations = []
        
        dc = _dynamic_control.acquire_dynamic_control_interface()
        articulation = dc.get_articulation(f"/World_{0}/franka")
        self.articulations.append(articulation)
        
        ########## let physics settle
        if simulation_context is not None:
            for _ in range(60):
                simulation_context.step(render=False)
     
            self.checker.initialization_step()

        # settle checker (we use checker to initialze articulation body states)
        if simulation_context is not None:
            for _ in range(10):
                simulation_context.step(render=False)
        
        self.time_step = 0
        ########## setup controller
        self.gripper_controller = self.robot.gripper
        self.c_controller = RMPFlowController(name="cspace_controller", robot_articulation=self.robot, physics_dt=1/120.0)

        return self.render()

    def step(self):
        raise NotImplementedError

    def _define_stage_properties(self):
        set_stage_up_axis(self.stage_properties.scene_up_axis)
        set_stage_units(self.stage_properties.scene_stage_unit)
        self._set_up_physics_secne()
        
        skylight_path = '/skylight'
        add_reference_to_stage(self.stage_properties.light_usd_path, skylight_path)

    def _set_up_physics_secne(self):
        # reference : https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_transfer_policy.html
        physicsScenePath = "/physicsScene"
        scene = UsdPhysics.Scene.Get(self.stage, physicsScenePath)
        if not scene:
            scene = UsdPhysics.Scene.Define(self.stage, physicsScenePath)
        
        gravityDirection = self.stage_properties.gravity_direction
        self._gravityDirection = Gf.Vec3f(gravityDirection[0], gravityDirection[1],  gravityDirection[2])

        scene.CreateGravityDirectionAttr().Set(self._gravityDirection)

        self._gravityMagnitude = self.stage_properties.gravity_magnitude
        scene.CreateGravityMagnitudeAttr().Set(self._gravityMagnitude)
        
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        physxSceneAPI.GetTimeStepsPerSecondAttr().Set(120)
        physxSceneAPI.CreateEnableGPUDynamicsAttr().Set(self.use_gpu_physics )
        physxSceneAPI.CreateEnableEnhancedDeterminismAttr().Set(True)
        physxSceneAPI.CreateEnableStabilizationAttr().Set(True)

        physxSceneAPI.GetGpuMaxRigidContactCountAttr().Set(524288)
        physxSceneAPI.GetGpuMaxRigidPatchCountAttr().Set(81920)
        physxSceneAPI.GetGpuFoundLostPairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(262144)
        physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuMaxSoftBodyContactsAttr().Set(1048576)
        physxSceneAPI.GetGpuMaxParticleContactsAttr().Set(1048576)
        # physxSceneAPI.GetGpuHeapCapacityAttr().Set(67108864)
        
    def render(self):
        if not self._sensor_initialized:
            return None
        
        sensor_types = [self.sensor_types] * self.num_envs * len(self.camera_paths)
        verify_sensor_inits = [False] * self.num_envs * len(self.camera_paths)
        wait_times =  [ 0.0 ] * self.num_envs * len(self.camera_paths)

        simulation_context = SimulationContext.instance()
        simulation_context.render()

        time.sleep(0.05)
        simulation_context.render()
        simulation_context.render()
        
        gts = list(map(SyntheticDataHelper.get_groundtruth, self.sd_helpers, sensor_types,
                       self.viewport_windows, verify_sensor_inits, wait_times))
        
        gts = { 'images': gts, 'semantic_id': self.sd_helpers[0].get_semantic_id_map() }
        
        return gts

    def clear(self):
        from pxr import Sdf, Usd
        index = 0
        house_prim_path = f"/World_{index}/house"
        prim_path = Sdf.Path(house_prim_path)
        prim: Usd.Prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            delete_prim(house_prim_path)
        
        # delete_prim('/physicsScene')
    
    def _load_scene(self):
        index = 0
        house_prim_path = f"/World_{index}/house"
        # print("house usd path: ", self.scene_parameters[index].usd_path)
        # while True:
        
        house_prim = add_reference_to_stage(self.scene_parameters[index].usd_path, house_prim_path)
        self._wait_for_loading()
        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")
          
        house_prim = XFormPrim(house_prim_path)
        # print(euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )
        house_prim.set_local_pose(np.array([0,0,0]) )
        # house_prim.set_local_pose(np.array([0,0,0]),  euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )

        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        #TODO 
        # somehow setting this is convexhull sometimes will lead to crash in omniverse
        setStaticCollider(furniture_prim, approximationShape=CONVEXHULL)
    
        self._wait_for_loading()

        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")

        #TODO 
        # somehow setting this to convexHull will lead to some bug need to modify meshes later
        setStaticCollider(room_struct_prim, approximationShape="none")

        floor_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].floor_path}")
        self._set_ground_plane(index)
        
        wall_material_url = self.scene_parameters[index].wall_material_url
        floor_material_url = self.scene_parameters[index].floor_material_url
        
        if wall_material_url and floor_material_url:
            #TODO
            # this needs some refactor 
            wall_mtl_name = wall_material_url.split("/")[-1][:-4]
            floor_mtl_name = floor_material_url.split("/")[-1][:-4]
            if wall_mtl_name not in BaseTask.material_library:
                _, wall_material_prim_path = get_material_prim_path(wall_mtl_name)
                BaseTask.material_library[wall_mtl_name] = wall_material_prim_path
            else:
                wall_material_prim_path = BaseTask.material_library[wall_mtl_name]
            
            if floor_mtl_name not in BaseTask.material_library:
                _, floor_material_prim_path = get_material_prim_path(floor_mtl_name)
                BaseTask.material_library[floor_mtl_name] = floor_material_prim_path
            else:
                floor_material_prim_path = BaseTask.material_library[floor_mtl_name]
            
            # print("floor_material_url: ", floor_material_url)
            if floor_material_prim_path:
                # self._assets_root_path = get_assets_root_path()
                # print("load floor material")
                omni.kit.commands.execute(
                    "CreateMdlMaterialPrim",
                    mtl_url=floor_material_url,
                    mtl_name=floor_mtl_name,
                    mtl_path=floor_material_prim_path,
                    select_new_prim=False,
                )
                self._wait_for_loading()
                # print("created floor material")
                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=floor_prim.GetPath(),
                    material_path=floor_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                self._wait_for_loading()
                # print("load floor material done")
            
            # print("wall_material_url: ", wall_material_url)
            if wall_material_prim_path:
                # print("load wall material")
                omni.kit.commands.execute(
                    "CreateMdlMaterialPrim",
                    mtl_url=wall_material_url,
                    mtl_name=wall_mtl_name,
                    mtl_path=wall_material_prim_path,
                    select_new_prim=False,
                )
                
                self._wait_for_loading()
                # print("created wall material")

                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=room_struct_prim.GetPath(),
                    material_path=wall_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                
                self._wait_for_loading()
                # print("load wall material done")
        
        self._wait_for_loading()

    def _set_ground_plane(self,index):
        ground_plane_path = f"/World_{index}/house/groundPlane"
        physicsUtils.add_ground_plane(self.stage,  ground_plane_path, "Y", 5000.0, 
            pxr.Gf.Vec3f(0.0, 0.0, 0.0), pxr.Gf.Vec3f(0.2))
        ground_prim = self.stage.GetPrimAtPath(ground_plane_path)
        #if self.is_loading_scene:
        ground_prim.GetAttribute('visibility').Set('invisible')

    def _load_robot(self):
        # using one environment for now
        index = 0
        prim_path = f"/World_{index}/franka"

        position = self.robot_parameters[index].robot_position
        rotation = self.robot_parameters[index].robot_orientation_quat
        
        # position, rotation = self._y_up_to_z_up(position=position, rotation=rotation)

        robot = Franka(
                prim_path = prim_path, name = f"my_frankabot{index}",
                usd_path = self.robot_parameters[index].usd_path,
                orientation = rotation,
                position = position,
                end_effector_prim_name = 'panda_rightfinger',
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
            )
        
        add_update_semantics(get_prim_at_path(prim_path), "Robot")
        self._wait_for_loading()
        self._set_sensors(robot)
     
        return robot

    def _set_sensors(self, robot=None):
        self._register_camera_path(robot)
        BaseTask.sd_helpers = [] 
        BaseTask.viewport_windows = []

        if len(BaseTask.viewport_handles) == 0:
            for idx, camera_path in enumerate(self.camera_paths):
                print("camera_path: ", camera_path)
                viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
                viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
                viewport_window.set_active_camera(camera_path)
                viewport_window.set_texture_resolution(*self.sensor_resolution)
                # viewport_window.set_window_pos(300*int(idx/2), 300 * int(idx%2))
                viewport_window.set_window_pos(1000, 400)
                viewport_window.set_window_size(300, 300)

                sd_helper = SyntheticDataHelper()
                sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                BaseTask.sd_helpers.append(sd_helper)
                BaseTask.viewport_windows.append(viewport_window)
                BaseTask.viewport_handles.append(viewport_handle)
        
        else:
            for viewport_handle, camera_path in zip( BaseTask.viewport_handles , self.camera_paths):
                viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
                viewport_window.set_active_camera(camera_path)
                viewport_window.set_texture_resolution(*self.sensor_resolution)
                viewport_window.set_window_pos(1000, 400)
                viewport_window.set_window_size(300, 300)

                BaseTask.viewport_windows.append(viewport_window)
                sd_helper = SyntheticDataHelper()
                sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                BaseTask.sd_helpers.append(sd_helper)
        
        self.kit.update()
        self._sensor_initialized = True
    
    def _register_camera_path(self, robot=None):
        self.camera_paths = []
        if robot is not None:
            self._add_cube_cameras()

        robot_path = f'/World_{0}/franka'
        camera_paths = [ 
            f'{robot_path}/FrontCamera', 
            f'{robot_path}/BaseCamera',
            f'{robot_path}/LeftCamera',
            f'{robot_path}/panda_hand/gripperCameraBottom',
            f'{robot_path}/panda_hand/gripperCamera',
        ]

        camera_paths.extend([
            "/World_0/Camera_Forward",
            "/World_0/Camera_Top",
            "/World_0/Camera_Back",
            "/World_0/Camera_Left",
            "/World_0/Camera_Right",
            ])
        
        for camera_path in camera_paths:
            self.camera_paths.append(camera_path)


    def _add_cube_cameras(self, usd_stage=None):
        # Access the main USD stage
        main_stage = omni.usd.get_context().get_stage()

        robot_path = "/World_0/franka" 
        franka_gripper_path = f"{robot_path}/panda_hand" 
        franka_base_path = robot_path

        # Get the transform for the franka_base
        franka_base = main_stage.GetPrimAtPath(franka_base_path)
        franka_grip = main_stage.GetPrimAtPath(franka_gripper_path)
        pose = omni.usd.utils.get_world_transform_matrix(franka_base)
        grip_pose = omni.usd.utils.get_world_transform_matrix(franka_grip)

        # Extract translation (translation part of the matrix)
        arnold_base_trans = pose.ExtractTranslation()
        arnold_base_rotation = pose.ExtractRotationMatrix()
        
        arnold_gripper_trans = grip_pose.ExtractTranslation()

        arnold_base_pose = np.eye(4, 4)
        arnold_base_pose[:3, :3] = arnold_base_rotation
        arnold_base_pose[:3, -1] = arnold_base_trans


        house_path = "/World_0/house"

        # Define the camera path within the main stage (you can choose any path you like)
        camera_path_forward = f"/World_0/Camera_Forward"
        camera_path_top = f"/World_0/Camera_Top"
        camera_path_back = f"/World_0/Camera_Back"
        camera_path_left = f"/World_0/Camera_Left"
        camera_path_right = f"/World_0/Camera_Right"
        cam_paths = [camera_path_forward, camera_path_top, camera_path_back, camera_path_left, camera_path_right]
        # ['forward', 'top', 'back', 'left', 'right']


        translations = np.array([
       [ 2.01308787e-01, -4.87864017e-05,  2.67479619e+00],
       [ 2.01308787e-01,  1.74974408e+00,  1.17497365e+00],
       [ 2.01308787e-01, -1.74984165e+00,  1.17497375e+00],
       [ 1.95110166e+00, -4.87864017e-05,  1.17497365e+00],
       [-1.54848408e+00, -4.87864017e-05,  1.17497375e+00]])

        base_translations = np.array([
       [ 2.30881   ,  0.012797  ,  0.27997002],
       [ 0.50881   ,  0.012797  ,  1.67997   ],
       [-1.19119   ,  0.012797  ,  0.27997002],
       [ 0.50881   , -1.78720295,  0.27997002],
       [ 0.50881   ,  1.81279695,  0.27997002]]) * 100
        
        transformations = np.array([
        [[-1.19209261e-07, -9.99999881e-01, -7.15255553e-07,  1.02519965e-06],
        [-7.10542482e-14, -5.96046306e-07,  9.99999881e-01, -1.09999989e+00],
        [-9.99999881e-01,  2.38418451e-07,  1.19209432e-07,  1.99999963e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],],
        [[ 4.76837134e-07, -1.00000000e+00, -2.05465780e-07,  4.18297021e-07],
        [-1.00000000e+00, -3.57627892e-07,  1.15903729e-07,  1.99999713e-01],
        [-1.15903815e-07,  2.05465740e-07, -1.00000000e+00,  2.50000002e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],],
        [[ 2.38418650e-07,  1.00000000e+00, -5.96046419e-07,  1.01327905e-06],
        [-1.19209147e-07,  5.96046476e-07,  1.00000000e+00, -1.10000020e+00],
        [ 1.00000000e+00, -2.38418508e-07,  1.19209432e-07,  1.49999987e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],],
        [[-1.00000000e+00,  1.15903826e-07, -5.74250541e-07,  2.00000843e-01],
        [-5.74250576e-07, -1.78813889e-07,  1.00000006e+00, -1.10000030e+00],
        [ 1.15903755e-07,  1.00000006e+00,  1.19209370e-07 , 1.79999991e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],],
        [[ 1.00000000e+00, -3.00296271e-07, -5.63713837e-07, -1.99998842e-01],
        [ 5.63713938e-07,  3.57627742e-07,  1.00000006e+00, -1.10000085e+00],
        [-3.00296066e-07, -1.00000006e+00,  3.57628081e-07,  1.79999973e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],],
        ])

        # subtract rlbench base from inv(exts)
        transformations[-1][:3, -1] -= [-0.012797, 0.82003, -0.30881]
        transformations[-2][:3, -1] -= [-0.012797, 0.82003, -0.30881]

        for i, original_pose_matrix in enumerate(transformations):
            # Extract the original rotation matrix (YZX) and translation vector
            original_rotation_matrix = original_pose_matrix[:3, :3]
            original_translation_vector = original_pose_matrix[:3, 3]

            # Create a rotation object from the original rotation matrix
            rot_YZX = R.from_matrix(original_rotation_matrix)

            # Convert the rotation to Euler angles in 'XYZ' order
            rot_XYZ = rot_YZX.as_euler('XYZ', degrees=False)

            # Create a new 4x4 'XYZ' pose matrix 
            xyz_pose_matrix = np.eye(4, 4)
            xyz_pose_matrix[:3, :3] = R.from_euler('XYZ', rot_XYZ, degrees=False).as_matrix()
            xyz_pose_matrix[:3, 3] = original_translation_vector[[2, 0, 1]]

            # overwrite the transformations
            transformations[i] = xyz_pose_matrix

        # lower z-value means going higher in arnold
        transformations[-1][2, -1] *= -1
        transformations[-2][2, -1] *= -1

        # convert to cm
        transformations[-1][:3, -1] *= 100
        transformations[-2][:3, -1] *= 100

        # transform to get left and right cube positions
        left_cam_transformed = arnold_base_pose @ transformations[-1] 
        left_cam_pos = left_cam_transformed[:3, -1]
        right_cam_transformed = arnold_base_pose @ transformations[-2]
        right_cam_pos = right_cam_transformed[:3, -1]

        print("BASE TRANS")
        print(arnold_base_trans)
        
        # This does not work correct for calculating COA!
        center_of_action = (left_cam_pos + right_cam_pos) / 2

        # This works well for COA
        # center_of_action = (arnold_gripper_trans + translations[-1]) + (arnold_gripper_trans + translations[-2])
        # center_of_action /= 2

        print("CENTER OF ACTION")
        print(center_of_action)
        
        translations = np.array([
            [0, -1.74979287, 0], # 0.2 is too close, the camera doesn't show the robot at all!
            [0, 0, 1.74979287], # 1.74979287
            [0, 1.74979287, 0],
            [-1.74979287, 0, 0],
            [1.74979287, 0, 0],
            ]) * 5 # make the cameras go further to have the whole image, otherwise the robot is not contained in images
        
        # The default metric in ommni kit is centimeters, so we multiply by 100
        translations *= 100

        for i, cam_path in enumerate(cam_paths):
            # Create the camera and set its attributes
            camera = UsdGeom.Camera.Define(main_stage, cam_path)
            camera = UsdGeom.Xformable(camera)

            # Calculate the up vector (you can choose an appropriate up vector based on your desired orientation)
            up_vector = Gf.Vec3d(0.0, 1.0, 0.0)

            cam_loc = Gf.Vec3d(*translations[i]) + Gf.Vec3d(*center_of_action)  # Camera position
            
            # Calculate the rotation matrix to look at the center of action
            rotation_matrix = Gf.Matrix4d().SetLookAt(
                cam_loc,
                Gf.Vec3d(*center_of_action),  # Look-at position
                up_vector  # Up vector
            )
            
            # rotation_matrix = Gf.Matrix3d(cam_pose[:3, :3].tolist())
            # rotation_matrix = Gf.Matrix3d(rotations[i])

            # Apply translation
            transformation_matrix = Gf.Matrix4d()
            transformation_matrix.SetTranslateOnly(cam_loc)
            transformation_matrix.SetRotateOnly(rotation_matrix.ExtractRotation())
            
            camera.ClearXformOpOrder()
            camera.AddTransformOp().Set(transformation_matrix)
            
            if 'right' in cam_path.lower() or 'left' in cam_path.lower():
                # Rotate the camera by 180 degrees around its local Y-axis
                camera.AddRotateYOp().Set(value=180.0)


            self._set_prim_invisible(house_path, main_stage)
            
            # print('Cam pose')
            # print(transformation_matrix)


    def _set_prim_invisible(self, prim_path, main_stage):
        prim = main_stage.GetPrimAtPath(prim_path)
        prim_visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()

        # Set the visibility to invisible for the cube cameras
        prim_visibility_attr.Set(value=UsdGeom.Tokens.invisible)

    def _wait_for_loading(self):
        sim = SimulationContext.instance()
        sim.render()
        while is_stage_loading():
            sim.render()