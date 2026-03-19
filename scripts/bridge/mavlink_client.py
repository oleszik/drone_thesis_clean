from __future__ import annotations


def connect_mavlink(connection: str):
    from pymavlink import mavutil

    mav = mavutil.mavlink_connection(connection)
    return mav, mavutil


def maybe_switch_to_sitl_master(connection: str, prefer_sitl_tcp: int, probe_timeout_s: float = 1.0) -> str:
    requested = str(connection).strip()
    if int(prefer_sitl_tcp) != 1:
        return requested
    if requested.lower() != "udp:127.0.0.1:14550":
        return requested
    sitl_master = "tcp:127.0.0.1:5760"
    print(f"[bridge] probing SITL master {sitl_master} for heartbeat (timeout={float(probe_timeout_s):.1f}s)")
    try:
        from pymavlink import mavutil

        probe = mavutil.mavlink_connection(sitl_master)
        try:
            probe.wait_heartbeat(timeout=float(probe_timeout_s))
            print("[bridge] switching to SITL master tcp:127.0.0.1:5760 for reliable mode/arming")
            return sitl_master
        finally:
            try:
                probe.close()
            except Exception:
                pass
    except Exception as exc:
        print(f"[bridge] keeping requested link {requested}; SITL master probe failed: {exc}")
        return requested
