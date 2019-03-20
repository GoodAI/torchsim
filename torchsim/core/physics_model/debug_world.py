import pygame
import pymunk.pygame_util
from pygame.color import THECOLORS

from torchsim.core.physics_model.latent_world import LatentWorld
from torchsim.core.physics_model.pymunk_physics import PyMunkPhysics, TemporalClass, Instance, InstanceColor, InstanceShape


class MyDrawOptions(pymunk.pygame_util.DrawOptions):
    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        super().draw_circle(pos, angle, radius, outline_color, fill_color)
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 8)
        textsurface = myfont.render(InstanceShape(255 - fill_color[3]).name, False, (0, 0, 0))
        self.surface.blit(textsurface, (pos[0], self.surface.get_height() - pos[1]))


class DebugWorld:

    def __init__(self, pymunk_physics: PyMunkPhysics):
        # pygame
        pygame.init()
        self.physics = pymunk_physics
        self._screen = pygame.display.set_mode((pymunk_physics.world_dims[0] + 1, pymunk_physics.world_dims[1] + 1))
        self._clock = pygame.time.Clock()

        self._draw_options = MyDrawOptions(self._screen)
        self._running = True

        # latent world code
        self.latent_world = LatentWorld()

    def show(self):
        self._clear_screen()
        self._draw_objects()
        pygame.display.flip()
        print(self.latent_world.to_tensor(self.physics.instances))


    def run(self):
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            self.physics.step()

            self._process_events()

            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            print(self.latent_world.to_tensor(physics.instances))
            # pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(THECOLORS["white"])

    def _draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        self._draw_options.DRAW_COLLISION_POINTS = False

        for object in self.physics.instances:
            object.pm_shape.color = (object.pm_shape.color[0], object.pm_shape.color[1], object.pm_shape.color[2],
                                     255 - object.shape.value)

        self.physics._space.debug_draw(self._draw_options)

    def _process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                pass
                # if event.key == pygame.K_SPACE:
                #     self.physics.add_object(PhysicObject(ObjectShape.CIRCLE, ObjectBehaviour.GRAVITY, size=15))
                #
                # if event.key == pygame.K_BACKSPACE:
                #     self.add_object(PhysicObject(ObjectShape.CIRCLE, ObjectBehaviour.BOUNCE, size=15))
                #
                # if event.key == pygame.K_ESCAPE:
                #     self.add_object(PhysicObject(ObjectShape.CIRCLE, ObjectBehaviour.GRAVITY, size=15))


if __name__ == '__main__':
    temporal_classes = [
        TemporalClass([
            Instance(100, init_position=(50, 100), init_direction=(20, 0))
        ]),
        TemporalClass([
            Instance(100, init_position=(100, 100), init_direction=(50, 20), color=InstanceColor.GREEN,
                     shape=InstanceShape.SQUARE),
            Instance(50, color=InstanceColor.BLUE, shape=InstanceShape.TRIANGLE)
        ]),
        TemporalClass([
            Instance(30, init_position=(100, 50), init_direction=(50, 20), color=InstanceColor.GREEN,
                     shape=InstanceShape.SQUARE),
            Instance(50, color=InstanceColor.BLUE, shape=InstanceShape.TRIANGLE)
        ])
    ]
    physics = PyMunkPhysics([150, 150], temporal_classes)

    game = DebugWorld(physics)

    game.run()



