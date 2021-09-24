import collections
import time
from abc import ABCMeta

import katagames_sdk.engine as kataen


pygame = kataen.import_pygame()
EventReceiver = kataen.EventReceiver
EngineEvTypes = kataen.EngineEvTypes


class BaseGame(metaclass=ABCMeta):
    """
    Base class for games, this is staged for the katasdk0.0.7 release
    """

    DEFAULT_FPS_N_FRAMES = 16

    def __init__(self, track_fps=False):
        if track_fps:
            self.__track_fps = True
            self.fps_n_frames = self.DEFAULT_FPS_N_FRAMES
            self.fps_tracker = collections.deque()
        else:
            self.__track_fps = False

        self.tick = 0

    @property
    def is_tracking_fps(self):
        return self.__track_fps

    def start(self):
        """Starts the game loop. This method will not exit until the game has finished execution."""

        class _ProxyReceiver(EventReceiver):
            def __init__(self, ref_game):
                super().__init__()

                self._game = ref_game
                self._event_queue = []
                self._last_update_time = time.time()

            def proc_event(self, ev, source):
                if ev.type == EngineEvTypes.LOGICUPDATE:
                    g = self._game
                    if g.is_tracking_fps:
                        g.fps_tracker.append(ev.curr_t)
                        if len(g.fps_tracker) > g.fps_n_frames:
                            g.fps_tracker.popleft()

                    g.update(self._event_queue, ev.curr_t-self._last_update_time)
                    g.tick += 1
                    self._last_update_time = ev.curr_t
                    self._event_queue.clear()

                elif ev.type == EngineEvTypes.PAINT:
                    self._game.render(ev.screen)
                else:
                    self._event_queue.append(ev)

        kataen.init(self.__get_mode_internal())
        b, a = kataen.get_game_ctrl(), _ProxyReceiver(self)
        a.turn_on()
        b.turn_on()

        self.pre_update()
        b.loop()
        kataen.cleanup()

    """
    implement methods below so the game class suits your needs
    """
    def get_mode(self) -> str:
        """
        you need to return either
        'HD', 'OLD_SCHOOL', or 'SUPER_RETRO'
        """
        raise NotImplementedError()

    def pre_update(self):
        pass

    def render(self, screen):
        raise NotImplementedError()

    def update(self, events, dt):
        raise NotImplementedError()

    """
    utils
    """
    def get_fps(self) -> float:
        if not self.__track_fps:
            raise ValueError('BaseGame has been built with a track_fps=False argument')
        else:
            q = self.fps_tracker
            total_time_secs = q[-1] - q[0]
            n_frames = len(q)
            if total_time_secs <= 0:
                return float('inf')
            else:
                return (n_frames - 1) / total_time_secs

    @staticmethod
    def get_screen_size():
        return kataen.get_screen().get_size()

    def get_tick(self) -> int:
        return self.tick

    @staticmethod
    def is_running_in_web() -> bool:
        return kataen.runs_in_web()

    """
    private methods
    """
    def __get_mode_internal(self):
        mode_str = self.get_mode().upper()
        if mode_str == 'HD':
            return kataen.HD_MODE
        elif mode_str == 'OLD_SCHOOL':
            return kataen.OLD_SCHOOL_MODE
        elif mode_str == 'SUPER_RETRO':
            return kataen.SUPER_RETRO_MODE
        else:
            raise ValueError("Unrecognized mode: {}".format(mode_str))
