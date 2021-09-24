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

    FPS_TRACKING_DEFAULT_SS = 13

    def __init__(self, track_fps=True):
        if track_fps:
            self._fps_n_frames = self.FPS_TRACKING_DEFAULT_SS
        else:
            self._fps_n_frames = 0

        self._fps_tracker = collections.deque()
        self._tick = 0

        self._cached_info_text = None
        self._info_font = None

    def start(self):
        """Starts the game loop. This method will not exit until the game has finished execution."""

        class _ProxyReceiver(EventReceiver):
            def __init__(self, ref_game):
                super().__init__()
                self._tnow_cache = None
                self._game = ref_game
                self._event_queue = []
                self._last_update_time = time.time()

            def proc_event(self, ev, source):
                if ev.type == EngineEvTypes.LOGICUPDATE:
                    self._game._update_internal(self._event_queue, ev.curr_t, ev.curr_t-self._last_update_time)
                    self._last_update_time = self._tnow_cache = ev.curr_t
                    self._event_queue.clear()
                if ev.type == EngineEvTypes.PAINT:
                    self._game._render_internal(ev.screen, self._tnow_cache)
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
        if len(self._fps_tracker) <= 1:
            return 0
        else:
            q = self._fps_tracker
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
        return self._tick

    @staticmethod
    def is_running_in_web() -> bool:
        return kataen.runs_in_web()

    def render_text(self, screen, text, size=12, pos=(0, 0), xanchor=0, color=(255, 255, 255), bg_color=None):
        if self._info_font is None or self._info_font.get_height() != size:
            self._info_font = pygame.font.Font(None, size)
        lines = text.split("\n")
        y = pos[1]
        for a_line in lines:
            surf = self._info_font.render(a_line, True, color, bg_color)
            screen.blit(surf, (int(pos[0] - xanchor * surf.get_width()), y))
            y += surf.get_height()

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

    def _update_internal(self, events, tnow, dt):
        self.update(events, dt)
        self._tick += 1

    def _render_internal(self, screen, tnow):
        if self._fps_n_frames > 0:
            self._fps_tracker.append(tnow)
            if len(self._fps_tracker) > self._fps_n_frames:
                self._fps_tracker.popleft()
        self.render(screen)
