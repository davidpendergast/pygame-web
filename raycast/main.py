
############## game.py ##############
# TODO web exports only support a single file right now (I think)


import katagames_sdk.engine as kataen

import time
import collections

pygame = kataen.import_pygame()
EventReceiver = kataen.EventReceiver
EngineEvTypes = kataen.EngineEvTypes


class Game:
    """Base class for games."""

    def __init__(self, track_fps=True):
        self._fps_n_frames = 10 if track_fps else 0
        self._fps_tracker_logic = collections.deque()
        self._fps_tracker_rendering = collections.deque()
        self._tick = 0

        self._cached_info_text = None
        self._info_font = None

    def start(self):
        """Starts the game loop. This method will not exit until the game has finished execution."""
        kataen.init(self._get_mode_internal())

        li_recv = [kataen.get_game_ctrl(), self.build_controller()]
        for recv_obj in li_recv:
            recv_obj.turn_on()

        self.pre_update()

        li_recv[0].loop()
        kataen.cleanup()

    def get_mode(self) -> str:
        """returns: "HD', 'OLD_SCHOOL', or 'SUPER_RETRO'"""
        return 'OLD_SCHOOL'

    def is_running_in_web(self) -> bool:
        return kataen.runs_in_web()

    def get_screen_size(self):
        return kataen.get_screen().get_size()

    def get_tick(self) -> int:
        return self._tick

    def pre_update(self):
        pass

    def render(self, screen):
        raise NotImplementedError()

    def update(self, events, dt):
        raise NotImplementedError()

    def render_text(self, screen, text, size=12, pos=(0, 0), color=(255, 255, 255), bg_color=None):
        if self._info_font is None or self._info_font.get_height() != size:
            self._info_font = pygame.font.Font(None, size)
        lines = text.split("\n")
        y = pos[1]
        for l in lines:
            surf = self._info_font.render(l, True, color, bg_color)
            screen.blit(surf, (pos[0], y))
            y += surf.get_height()

    def get_fps(self, logical=True) -> float:
        q = self._fps_tracker_logic if logical else self._fps_tracker_rendering
        if len(q) <= 1:
            return 0
        else:
            total_time_secs = q[-1] - q[0]
            n_frames = len(q)
            if total_time_secs <= 0:
                return float('inf')
            else:
                return (n_frames - 1) / total_time_secs

    def _render_internal(self, screen):
        if self._fps_n_frames > 0:
            self._fps_tracker_rendering.append(time.time())
            if len(self._fps_tracker_rendering) > self._fps_n_frames:
                self._fps_tracker_rendering.popleft()
        self.render(screen)

    def _update_internal(self, events, dt):
        if self._fps_n_frames > 0:
            self._fps_tracker_logic.append(time.time())
            if len(self._fps_tracker_logic) > self._fps_n_frames:
                self._fps_tracker_logic.popleft()
        self.update(events, dt)
        self._tick += 1

    def _get_mode_internal(self):
        mode_str = self.get_mode().upper()
        if mode_str == 'HD':
            return kataen.HD_MODE
        elif mode_str == 'OLD_SCHOOL':
            return kataen.OLD_SCHOOL_MODE
        elif mode_str == 'SUPER_RETRO':
            return kataen.SUPER_RETRO_MODE
        else:
            raise ValueError("Unrecognized mode: {}".format(mode_str))

    class _GameViewController(EventReceiver):
        def __init__(self, game):
            super().__init__()
            self._game = game
            self._event_queue = []
            self._last_update_time = time.time()

        def proc_event(self, ev, source):
            if ev.type == EngineEvTypes.PAINT:
                self._game._render_internal(ev.screen)
            elif ev.type == EngineEvTypes.LOGICUPDATE:
                cur_time = time.time()
                self._game._update_internal(self._event_queue, cur_time - self._last_update_time)
                self._last_update_time = cur_time
                self._event_queue.clear()
            else:
                self._event_queue.append(ev)

    def build_controller(self) -> EventReceiver:
        return Game._GameViewController(self)


############## game.py ##############

############## raycaster.py ##############

import math


class Vector2:
    # pygame.Vector2 doesn't seem to be supported yet
    # So I'll make my own >:(

    def __init__(self, x, y=0.0):
        if isinstance(x, Vector2):
            self.x = x.x
            self.y = x.y
        else:
            self.x = x
            self.y = y

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        else:
            return self.y

    def __len__(self):
        return 2

    def __iter__(self):
        return (v for v in (self.x, self.y))

    def __add__(self, other: 'Vector2'):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2'):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vector2(self.x * other, self.y * other)

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: 'Vector2'):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def rotate_ip(self, degrees):
        theta = math.radians(degrees)
        cs = math.cos(theta)
        sn = math.sin(theta)
        x = self.x * cs - self.y * sn
        y = self.x * sn + self.y * cs
        self.x = x
        self.y = y

    def rotate(self, degrees):
        res = Vector2(self)
        res.rotate_ip(degrees)
        return res

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def length_squared(self):
        return self.x * self.x + self.y * self.y

    def scale_to_length(self, length):
        cur_length = self.length()
        if cur_length == 0 and length != 0:
            raise ValueError("Cannot scale vector with length 0")
        else:
            mult = length / cur_length
            self.x *= mult
            self.y *= mult


class RayEmitter:

    def __init__(self, xy, direction, fov, n_rays, max_depth=100):
        self.xy = xy
        self.direction = direction
        self.fov = fov
        self.n_rays = max(n_rays, 3)
        self.max_depth = max_depth

    def get_rays(self):
        left_ray = self.direction.rotate(-self.fov / 2)
        for i in range(self.n_rays):
            yield left_ray.rotate((i + 0.5) * self.fov / self.n_rays)


class RayCastPlayer(RayEmitter):

    def __init__(self, xy, direction, fov, n_rays):
        super().__init__(xy, direction, fov, n_rays)
        self.move_speed = 50  # units per second
        self.turn_speed = 120

    def move(self, forward, strafe, dt):
        if forward != 0:
            self.xy = self.xy + self.direction * forward * self.move_speed * dt

        if strafe != 0:
            right = self.direction.rotate(90)
            self.xy = self.xy + right * strafe * self.move_speed * dt

    def turn(self, direction, dt):
        self.direction.rotate_ip(direction * self.turn_speed * dt)


class RayCastWorld:

    def __init__(self, grid_size, bg_color=(0, 0, 0)):
        self.grid = []
        for _ in range(grid_size[1]):
            self.grid.append([None] * grid_size[0])
        self.bg_color = bg_color

    def get_size(self):
        if len(self.grid) == 0:
            return (0, 0)
        else:
            return (len(self.grid[0]), len(self.grid))

    def get_width(self):
        return self.get_size()[0]

    def get_height(self):
        return self.get_size()[1]


class RayCastState:

    def __init__(self, player: RayCastPlayer, world: RayCastWorld):
        self.player = player
        self.world = world


class RayCastRenderer:

    def __init__(self):
        pass

    def render(self, screen, state: RayCastState):
        rays = [r for r in state.player.get_rays()]
        xy = state.player.xy
        for r in rays:
            pygame.draw.line(screen, (255, 255, 255), xy, xy + r * 100)


class RayCasterGame(Game):

    def __init__(self):
        super().__init__()
        self.state = None
        self.renderer = RayCastRenderer()
        self.show_fps = True

    def _build_initial_state(self):
        w = RayCastWorld(self.get_screen_size())
        p = RayCastPlayer(Vector2(w.get_width() / 2, w.get_height() / 2),
                          Vector2(0, 1),
                          60, 20)
        return RayCastState(p, w)

    def get_mode(self):
        return 'SUPER_RETRO'

    def update(self, events, dt):
        if self.state is None:
            self.state = self._build_initial_state()
        if self.get_tick() % 20 == 0:
            dims = self.get_screen_size()
            cap = "Raycaster (DIMS={}, FPS={:.1f})".format(dims, self.get_fps(logical=False))
            pygame.display.set_caption(cap)

        pressed = pygame.key.get_pressed()

        turn = 0
        if pressed[pygame.K_q] or pressed[pygame.K_LEFT]:
            turn -= 1
        if pressed[pygame.K_e] or pressed[pygame.K_RIGHT]:
            turn += 1

        forward = 0
        if pressed[pygame.K_w] or pressed[pygame.K_UP]:
            forward += 1
        if pressed[pygame.K_s] or pressed[pygame.K_DOWN]:
            forward -= 1

        strafe = 0
        if pressed[pygame.K_a]:
            strafe -= 1
        if pressed[pygame.K_d]:
            strafe += 1

        self.state.player.turn(turn, dt)
        self.state.player.move(forward, strafe, dt)

    def render(self, screen):
        screen.fill((0, 0, 0))
        self.renderer.render(screen, self.state)

        if self.show_fps:
            fps_text = "FPS {:.1f}".format(self.get_fps(logical=False))
            self.render_text(screen, fps_text, bg_color=(0, 0, 0), size=16)

############## raycaster.py ##############

############## main.py ##############


def run_game():
    """Entry point for packaged web runs"""
    g = RayCasterGame()
    g.start()


if __name__ == '__main__':
    """Entry point for offline runs"""
    run_game()

############## main.py ##############
