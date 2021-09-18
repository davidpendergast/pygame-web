
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

    def get_tick(self) -> int:
        return self._tick

    def pre_update(self):
        pass

    def render(self, screen):
        raise NotImplementedError()

    def update(self, events, dt):
        raise NotImplementedError()

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


class RayCasterGame(Game):

    def __init__(self):
        super().__init__()

    def get_mode(self):
        return 'SUPER_RETRO'

    def update(self, events, dt):
        if self.get_tick() % 20 == 0:
            dims = kataen.get_screen().get_size()
            cap = "Raycaster (FPS={:.1f}, DIMS={})".format(self.get_fps(logical=False), dims)
            pygame.display.set_caption(cap)

    def render(self, screen):
        pass

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
