# import copy
#
# import pygame
# import pymunk
# import random
#
# from enum import Enum
# from typing import List
#
# from pygame.color import *
# from pymunk import Body
# import pymunk.pygame_util
#
#
# class ObjectBehaviour(Enum):
#     STATIC = 0
#     BOUNCE = 1
#     GRAVITY = 2
#     BOUNDS_FREE = 4
#
#
# class ObjectShape(Enum):
#     CIRCLE = 0
#     SQUARE = 1
#     TRIANGLE = 2
#
#
# class ObjectSpawnRule(Enum):
#     RANDOM = 0
#     TOP_OF_SCREEN = 1
#
#
# class PhysicObject:
#     size: int
#     direction: [int, int]
#     pos: [int, int]
#     shape: ObjectShape
#     behaviour: ObjectBehaviour
#
#     _body: pymunk.body
#     _shape: pymunk.shapes
#
#     def __init__(self, shape: ObjectShape, behaviour: ObjectBehaviour, size: int):
#         self.size = size
#         self.shape = shape
#         self.behaviour = behaviour
#         self.direction = [1, 1]
#
#
# class PhysicsModel:
#     _space: pymunk.space
#     _objects: List[PhysicObject]
#     _world_dims: [int, int]
#     _spawn_rule: ObjectSpawnRule
#
#     def __init__(self, world_dims: [int, int], spawn_rule: ObjectSpawnRule):
#         self._world_dims = world_dims
#         self._spawn_rule = spawn_rule
#         self._objects = []
#
#         self._init_scene()
#
#     def _init_scene(self):
#         # Space
#         self._space = pymunk.Space()
#         self._space.gravity = (0.0, -100.0)
#
#         # Physics
#         # Time step
#         self._dt = 1.0 / 60.0
#         # Number of physics steps per screen frame
#         self._physics_steps_per_frame = 1
#
#         # pygame
#         pygame.init()
#         self._screen = pygame.display.set_mode((self._world_dims[0] + 1, self._world_dims[1] + 1))
#         self._clock = pygame.time.Clock()
#
#         self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
#
#         # Static barrier walls (lines) that the balls bounce off of
#         self._add_world_bounds()
#
#         self._running = True
#
#     def _add_world_bounds(self):
#         """
#         Create the static bodies.
#         :return: None
#         """
#         static_body = self._space.static_body
#         width = self._world_dims[0] - 10
#         height = self._world_dims[1] - 10
#         static_lines = [pymunk.Segment(static_body, (10, 10), (width, 10), 0.0),
#                         pymunk.Segment(static_body, (width, 10), (width, height), 0.0),
#                         pymunk.Segment(static_body, (width, height), (10, height), 0.0),
#                         pymunk.Segment(static_body, (10, height), (10, 10), 0.0)]
#
#         for line in static_lines:
#             line.elasticity = 0.95
#             line.friction = 0.9
#
#         self._space.add(static_lines)
#
#     def add_object(self, obj: PhysicObject):
#         # generate position
#         pos = [0, 0]
#         if self._spawn_rule == ObjectSpawnRule.RANDOM:
#             x = random.uniform(0, self._world_dims[0])
#             y = random.uniform(0, self._world_dims[1])
#             pos = [x, y]
#
#         if self._spawn_rule == ObjectSpawnRule.TOP_OF_SCREEN:
#             x = random.uniform(0, self._world_dims[0])
#             y = self._world_dims[1] - 50
#             pos = [x, y]
#
#         # determine body type
#         if obj.behaviour == ObjectBehaviour.STATIC:
#             body_type = Body.STATIC
#         elif obj.behaviour == ObjectBehaviour.GRAVITY:
#             body_type = Body.DYNAMIC
#         else:
#             body_type = Body.KINEMATIC
#
#         # generate objects based on their shape
#         body: pymunk.body
#         shape: pymunk.shapes
#         if obj.shape == ObjectShape.CIRCLE:
#             body, shape = self._create_ball(obj.size, pos, body_type)
#         if obj.shape == ObjectShape.SQUARE:
#             body, shape = self._create_square(obj.size, pos, body_type)
#         if obj.shape == ObjectShape.TRIANGLE:
#             body, shape = self._create_triange(obj.size, pos, body_type)
#
#         # object reference
#         obj.pos = pos
#         obj._body = body
#         obj._shape = shape
#         self._objects.append(obj)
#         self._space.add(body, shape)
#
#     def replace_object(self, old: PhysicObject, new: PhysicObject):
#         for n, element in enumerate(self._objects):
#             if element == old:
#                 self._objects[n] = new
#
#     def remove_object(self, obj: PhysicObject):
#         self._space.remove(obj._body, obj._shape)
#         self._objects.remove(obj)
#
#     def _create_ball(self, size: int, pos: [int, int], body_type: Body.body_type):
#         """
#         Create a ball.
#         """
#
#         mass = 10
#         inertia = pymunk.moment_for_circle(mass, 0, size, (0, 0))
#         body = pymunk.Body(mass, inertia, body_type)
#         body.position = [pos[0], pos[1]]
#
#         shape = pymunk.Circle(body, size, (0, 0))
#         shape.elasticity = 0.95
#         shape.friction = 0.9
#         shape.velocity = 2
#
#         return body, shape
#
#     def _create_square(self, size: int, pos: [int, int], body_type: Body.body_type):
#         """
#         Create a square.
#         """
#
#         mass = 10
#         inertia = pymunk.moment_for_box(mass, (size, size))
#         body = pymunk.Body(mass, inertia, body_type)
#         body.position = [pos[0], pos[1]]
#
#         shape = pymunk.Poly.create_box(body, (size,size))
#         shape.elasticity = 0.95
#         shape.friction = 0.9
#
#         return body, shape
#
#     def _create_triange(self, size: int, pos: [int, int], body_type: Body.body_type):
#         """
#         Create a triangle.
#         """
#
#         mass = 10
#         vertices: pymunk.vec2d = {{-size / 2, -size / 2},
#                                   {+size / 2, +size / 2},
#                                   {+size / 2, -size / 2}}
#
#         inertia = pymunk.moment_for_poly(mass, 3, vertices, (0, 0))
#         body = pymunk.Body(mass, inertia, body_type)
#         body.position = [pos[0], pos[1]]
#
#         shape = pymunk.Poly(body, vertices)
#         shape.elasticity = 0.95
#         shape.friction = 0.9
#
#         return body, shape
#
#     def _update_scene(self):
#         """
#         Create/remove balls as necessary. Call once per frame only.
#         :return: None
#         """
#
#         for obj in self._objects:
#             if obj.behaviour == ObjectBehaviour.BOUNDS_FREE:
#                 # remove objects that get outside of world dimensions
#                 if obj.pos[0] not in range(0, self._world_dims[0]) or obj.pos[1] not in range(0, self._world_dims[1]):
#                     self.remove_object(obj)
#
#             if obj.behaviour == ObjectBehaviour.BOUNCE:
#                 x = obj.pos[0] + obj.direction[0]
#                 y = obj.pos[1] + obj.direction[1]
#
#                 if x < 20:
#                     obj.direction[0] = 1
#                 elif x > self._world_dims[0] - 20:
#                     obj.direction[0] = -1
#                 elif y < 20:
#                     obj.direction[1] = 1
#                 elif y > self._world_dims[1] - 20:
#                     obj.direction[1] = -1
#
#                 obj.pos = [x, y]
#                 obj._body.position = x, y
#
#     def _clear_screen(self):
#         """
#         Clears the screen.
#         :return: None
#         """
#         self._screen.fill(THECOLORS["white"])
#
#     def _process_events(self):
#         """
#         Handle game and events like keyboard input. Call once per frame only.
#         :return: None
#         """
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self._running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     self.add_object(PhysicObject(ObjectShape.SQUARE, ObjectBehaviour.GRAVITY, size=15))
#
#                 if event.key == pygame.K_BACKSPACE:
#                     self.add_object(PhysicObject(ObjectShape.SQUARE, ObjectBehaviour.BOUNCE, size=15))
#
#                 if event.key == pygame.K_ESCAPE:
#                     self.add_object(PhysicObject(ObjectShape.SQUARE, ObjectBehaviour.STATIC, size=15))
#
#     def _draw_objects(self):
#         """
#         Draw the objects.
#         :return: None
#         """
#         self._space.debug_draw(self._draw_options)
#
#     def run(self):
#         """
#         The main loop of the game.
#         :return: None
#         """
#         # Main loop
#         while self._running:
#             # Progress time forward
#             for x in range(self._physics_steps_per_frame):
#                 self._space.step(self._dt)
#
#             self._process_events()
#             self._update_scene()
#             self._clear_screen()
#             self._draw_objects()
#             pygame.display.flip()
#             # Delay fixed time between frames
#             self._clock.tick(50)
#             # pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
#
#
# if __name__ == '__main__':
#     game = PhysicsModel([600, 600], ObjectSpawnRule.RANDOM)
#     game.run()
