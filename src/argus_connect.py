"""Phone pairing for the A.R.G.U.S. server.

The phone apps (Android, iPhone) are thin companions: they send voice and camera frames to this PC
over the local network and show / speak the answers. Models only run on the PC. This module is what
lets a phone find the PC:

  * lan_addresses()      the PC's IPv4 addresses on Wi-Fi / hotspot / Ethernet, best guess first
  * connect_info(port)   everything the "Connect phone" panel shows, plus the pairing link
  * qr_svg(text)         the pairing link as a QR code (needs the small pure-Python `segno` package)
  * keep_advertising()   announces the server on the network (Bonjour / mDNS, `_argus._tcp`) so the
                         phone app lists this PC by itself (needs the `zeroconf` package)

Both extra packages are optional: without them the panel still shows the address to type in.

Pairing link:  argus://connect?h=<ip>[,<ip>...]&p=<port>&n=<pc name>
The phone tries each address in order and uses  ws://<ip>:<port>/ws.
"""
import ipaddress
import socket
import urllib.parse

SERVICE_TYPE = "_argus._tcp.local."
WS_PATH = "/ws"

try:
    import segno
except Exception:  # optional
    segno = None

try:
    from zeroconf import ServiceInfo, Zeroconf
except Exception:  # optional
    ServiceInfo = Zeroconf = None

try:
    import ifaddr
except Exception:  # comes with zeroconf
    ifaddr = None

_zeroconf = None
_service = None

# Adapters that are never the way to a phone (virtual machines, VPNs, WSL, Docker).
_VIRTUAL_HINTS = ("virtual", "vmware", "vbox", "hyper-v", "vethernet", "wsl", "docker", "loopback",
                  "tailscale", "zerotier", "tap-", "tun", "bluetooth", "npcap", "br-", "veth")


def pc_name():
    try:
        return socket.gethostname() or "ARGUS PC"
    except Exception:
        return "ARGUS PC"


def _usable(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and not (addr.is_loopback or addr.is_link_local or addr.is_multicast
                                      or addr.is_unspecified)


def _default_route_ip():
    """The address this PC uses to reach other machines. No packet is sent."""
    for target in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((target, 1))
            ip = s.getsockname()[0]
            if _usable(ip):
                return ip
        except OSError:
            pass
        finally:
            s.close()
    return None


def lan_addresses():
    """IPv4 addresses a phone on the same network could reach, most likely first."""
    found = []  # (rank, ip): lower rank first

    def add(ip, rank):
        if _usable(ip) and ip not in [f[1] for f in found]:
            found.append((rank, ip))

    primary = _default_route_ip()
    if primary:
        add(primary, 0)
    if ifaddr is not None:
        try:
            for adapter in ifaddr.get_adapters():
                label = f"{adapter.nice_name} {adapter.name}".lower()
                virtual = any(h in label for h in _VIRTUAL_HINTS)
                for ip in adapter.ips:
                    if isinstance(ip.ip, str):  # IPv4 (IPv6 is a tuple)
                        private = ipaddress.ip_address(ip.ip).is_private if _usable(ip.ip) else False
                        add(ip.ip, 3 if virtual else (1 if private else 2))
        except Exception:
            pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(ip, 2)
    except OSError:
        pass
    return [ip for _, ip in sorted(found, key=lambda f: f[0])]


def pairing_link(addresses, port, name=None):
    query = {"h": ",".join(addresses), "p": str(port), "n": name or pc_name()}
    return "argus://connect?" + urllib.parse.urlencode(query, safe=",")


def connect_info(port):
    addresses = lan_addresses()
    return {
        "name": pc_name(),
        "port": port,
        "addresses": addresses,
        "ws_urls": [f"ws://{ip}:{port}{WS_PATH}" for ip in addresses],
        "link": pairing_link(addresses, port) if addresses else "",
        "qr": bool(segno) and bool(addresses),
        "discovery": _zeroconf is not None,
    }


def qr_svg(text, dark="#111418", light="#ffffff"):
    """QR code as an SVG string, or None when `segno` isn't installed."""
    if segno is None or not text:
        return None
    import io
    out = io.BytesIO()
    segno.make(text, error="m").save(out, kind="svg", scale=8, border=3, dark=dark, light=light)
    return out.getvalue().decode("utf-8")


def start_advertising(port):
    """Announce the server on the local network so phone apps can list it. Returns True when on."""
    global _zeroconf, _service
    if Zeroconf is None or _zeroconf is not None:
        return _zeroconf is not None
    addresses = lan_addresses()
    if not addresses:
        return False
    try:
        name = pc_name()
        _service = ServiceInfo(
            SERVICE_TYPE,
            f"ARGUS on {name}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip) for ip in addresses],
            port=port,
            properties={"hosts": ",".join(addresses), "port": str(port), "path": WS_PATH, "name": name, "v": "1"},
            server=f"argus-{name.lower().replace(' ', '-').replace('.', '-')}.local.",
        )
        _zeroconf = Zeroconf()
        _zeroconf.register_service(_service, allow_name_change=True)
        print(f"[CONNECT] Phones can find this PC as \"ARGUS on {name}\" ({', '.join(addresses)}:{port})")
        return True
    except Exception as exc:
        print(f"[CONNECT] Network discovery is off ({type(exc).__name__}: {exc}). "
              f"Phones can still connect with the QR code or the address.")
        stop_advertising()
        return False


def stop_advertising():
    global _zeroconf, _service
    zc, _zeroconf = _zeroconf, None
    service, _service = _service, None
    if zc is not None:
        try:
            if service is not None:
                zc.unregister_service(service)
            zc.close()
        except Exception:
            pass


def keep_advertising(port, stop_event=None, every=10.0):
    """Run in a background thread: advertise, and re-advertise when the PC changes network
    (for example when it joins the phone's hotspot after ARGUS has started)."""
    import threading
    stop_event = stop_event or threading.Event()
    current = None
    while not stop_event.is_set():
        addresses = lan_addresses()
        if addresses != current:
            stop_advertising()
            current = addresses if start_advertising(port) else None
            if Zeroconf is None:
                return
        stop_event.wait(every)
    stop_advertising()


if __name__ == "__main__":
    import json
    print(json.dumps(connect_info(8000), indent=2))
