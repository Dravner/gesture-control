"""macOS input adapter. No arbitrary command execution; Accessibility remains OS-controlled."""
import sys
import ctypes
import numpy as np

def screen_position(x,y,width,height,origin=(0.,0.)):
    return float(origin[0]+np.clip(x,0,1)*(width-1)),float(origin[1]+np.clip(y,0,1)*(height-1))

# macOS virtual key codes (letters use the standard physical US positions).
KEYCODES=dict(zip('asdfhgzxcvbqwerty123465=97-80]ou[ip\rlj\'k;\\,/nm.',
    [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]))
KEYCODES.update({'space':49,'enter':36,'return':36,'tab':48,'backspace':51,'delete':51,'escape':53,'esc':53,'left':123,'right':124,'down':125,'up':126,
                'f1':122,'f2':120,'f3':99,'f4':118,'f5':96,'f6':97,'f7':98,'f8':100,'f9':101,'f10':109,'f11':103,'f12':111})
MODIFIERS={'command':1<<20,'cmd':1<<20,'shift':1<<17,'control':1<<18,'ctrl':1<<18,'alt':1<<19,'option':1<<19}

class MacActions:
    def __init__(self,display_id=None):
        self.display_id=display_id
        self._drag=False;self._position=(0.,0.)
        try:
            import Quartz
            self.q=Quartz
        except ImportError:self.q=None

    def available(self):
        if sys.platform!='darwin' or not self.q:return False
        try:
            api=ctypes.CDLL('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
            check=api.AXIsProcessTrusted;check.restype=ctypes.c_bool
            return bool(check())
        except (OSError,AttributeError):return False

    def set_display(self,display_id):
        self.release();self.display_id=int(display_id)

    def _screen_position(self,x,y):
        bounds=self.q.CGDisplayBounds(self.display_id if self.display_id is not None else self.q.CGMainDisplayID())
        origin=getattr(bounds,'origin',None)
        offset=(origin.x,origin.y) if origin is not None else (0.,0.)
        return screen_position(x,y,bounds.size.width,bounds.size.height,offset)

    def current_position(self):
        """Read cursor in the selected display's logical coordinates; never post input."""
        if self.q is None:raise RuntimeError('Quartz unavailable')
        bounds=self.q.CGDisplayBounds(self.display_id if self.display_id is not None else self.q.CGMainDisplayID())
        position=self.q.CGEventGetLocation(self.q.CGEventCreate(None))
        origin=getattr(bounds,'origin',None)
        offset=np.array([origin.x,origin.y]) if origin is not None else np.zeros(2)
        dimensions=np.array([bounds.size.width,bounds.size.height],dtype=float)
        raw=np.array([position.x,position.y],dtype=float)
        if not np.isfinite(np.r_[offset,dimensions,raw]).all() or np.any(dimensions<=1):raise ValueError('Invalid display/cursor coordinates')
        normalized=(raw-offset)/(dimensions-1)
        if np.any(normalized<0) or np.any(normalized>1):raise ValueError('Cursor is outside the selected display')
        return normalized

    def execute(self,event):
        if not self.available():raise PermissionError('Разрешите доступ Универсального доступа для приложения запуска в настройках macOS')
        q=self.q;action=event.action
        if action in {'none','pause'}:return
        if action=='move':
            self._position=self._screen_position(event.payload.get('x',.5),event.payload.get('y',.5))
            typ=q.kCGEventLeftMouseDragged if self._drag else q.kCGEventMouseMoved
            q.CGEventPost(q.kCGHIDEventTap,q.CGEventCreateMouseEvent(None,typ,self._position,q.kCGMouseButtonLeft))
        elif action in {'click','right_click','drag_start','drag_end'}:
            if action=='drag_start' and self._drag:return
            if action in {'click','right_click'} and self._drag:self.release()
            if action!='drag_end' and 'x' in event.payload and 'y' in event.payload:
                self._position=self._screen_position(event.payload['x'],event.payload['y'])
            else:
                pos=q.CGEventGetLocation(q.CGEventCreate(None));self._position=(pos.x,pos.y)
            right=action=='right_click';button=q.kCGMouseButtonRight if right else q.kCGMouseButtonLeft
            down=q.kCGEventRightMouseDown if right else q.kCGEventLeftMouseDown
            up=q.kCGEventRightMouseUp if right else q.kCGEventLeftMouseUp
            count=int(np.clip(event.payload.get('count',1),1,2)) if action in {'click','right_click'} else 1
            for index in range(count):
                for typ in ([up] if action=='drag_end' else [down] if action=='drag_start' else [down,up]):
                    mouse=q.CGEventCreateMouseEvent(None,typ,self._position,button)
                    if action in {'click','right_click'}:q.CGEventSetIntegerValueField(mouse,q.kCGMouseEventClickState,index+1)
                    q.CGEventPost(q.kCGHIDEventTap,mouse)
            self._drag=action=='drag_start'
        elif action=='scroll':
            if event.payload.get('unit')=='pixel':
                remainder=getattr(self,'_pixel_scroll_remainder',0.)+float(np.clip(event.payload.get('dy',0),-80,80))
                dy=int(np.trunc(remainder+np.sign(remainder)*1e-9));self._pixel_scroll_remainder=remainder-dy
                if dy:
                    wheel=q.CGEventCreateScrollWheelEvent(None,q.kCGScrollEventUnitPixel,1,dy)
                    q.CGEventSetIntegerValueField(wheel,q.kCGScrollWheelEventIsContinuous,1)
                    q.CGEventPost(q.kCGHIDEventTap,wheel)
            else:
                dy=int(np.clip(event.payload.get('dy',6),-30,30))
                if dy:q.CGEventPost(q.kCGHIDEventTap,q.CGEventCreateScrollWheelEvent(None,q.kCGScrollEventUnitLine,1,dy))
        elif action=='hotkey':
            keys=[x.lower() for x in event.payload.get('keys',[])];flags=0
            for key in keys:flags|=MODIFIERS.get(key,0)
            normal=[x for x in keys if x not in MODIFIERS]
            if any(x not in KEYCODES for x in normal):raise ValueError('Неподдерживаемая горячая клавиша')
            for key in normal:
                event_down=q.CGEventCreateKeyboardEvent(None,KEYCODES[key],True);q.CGEventSetFlags(event_down,flags);q.CGEventPost(q.kCGHIDEventTap,event_down)
                event_up=q.CGEventCreateKeyboardEvent(None,KEYCODES[key],False);q.CGEventSetFlags(event_up,flags);q.CGEventPost(q.kCGHIDEventTap,event_up)
        else:raise ValueError('Неизвестное действие')

    def release(self):
        self._pixel_scroll_remainder=0.
        if self._drag and self.q:
            q=self.q;q.CGEventPost(q.kCGHIDEventTap,q.CGEventCreateMouseEvent(None,q.kCGEventLeftMouseUp,self._position,q.kCGMouseButtonLeft))
        self._drag=False
