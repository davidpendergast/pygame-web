import katagames_sdk.engine as kataen

import time
import collections

pygame = kataen.import_pygame()
EventReceiver = kataen.EventReceiver
EngineEvTypes = kataen.EngineEvTypes
scr_size = None


class Game:

    def __init__(self, track_fps=True):
        self._fps_n_frames = 10 if track_fps else 0
        self._fps_tracker_logic = collections.deque()
        self._fps_tracker_rendering = collections.deque()

    def start(self):
        kataen.init(self.get_mode())

        li_recv = [kataen.get_game_ctrl(), GameViewController(self)]
        for recv_obj in li_recv:
            recv_obj.turn_on()

        self.pre_update()

        li_recv[0].loop()
        kataen.cleanup()

    def render_internal(self, screen):
        if self._fps_n_frames > 0:
            self._fps_tracker_rendering.append(time.time())
            if len(self._fps_tracker_rendering) > self._fps_n_frames:
                self._fps_tracker_rendering.popleft()
        self.render(screen)

    def update_internal(self, events, dt):
        if self._fps_n_frames > 0:
            self._fps_tracker_logic.append(time.time())
            if len(self._fps_tracker_logic) > self._fps_n_frames:
                self._fps_tracker_logic.popleft()
        self.update(events, dt)

    def get_mode(self):
        """returns: kataen.HD_MODE, kataen.OLD_SCHOOL_MODE, or kataen.SUPER_RETRO_MODE"""
        return kataen.OLD_SCHOOL_MODE

    def is_running_in_web(self):
        return kataen.runs_in_web()

    def pre_update(self):
        pass

    def render(self, screen):
        screen.fill(pygame.color.Color('antiquewhite2'))
        pygame.draw.circle(screen, (244, 105, 251), [240, 135], 15, 0)

    def update(self, events, dt):
        print("LOGIC_FPS={}, RENDER_FPS={}".format(self.get_fps(logical=True), self.get_fps(logical=False)))

    def get_fps(self, logical=True):
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


class GameViewController(EventReceiver):

    def __init__(self, game):
        super().__init__()
        self._game = game
        self._event_queue = []
        self._last_update_time = time.time()

    def proc_event(self, ev, source):
        if ev.type == EngineEvTypes.PAINT:
            self._game.render_internal(ev.screen)
        elif ev.type == EngineEvTypes.LOGICUPDATE:
            cur_time = time.time()
            self._game.update_internal(self._event_queue, cur_time - self._last_update_time)
            self._last_update_time = cur_time
            self._event_queue.clear()
        else:
            self._event_queue.append(ev)


# entry point for packaged web game
def run_game():
    game = Game()
    game.start()


# entry point for offline runs
if __name__ == '__main__':
    run_game()
    