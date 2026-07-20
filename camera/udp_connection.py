import ffmpeg
import numpy as np
import cv2
import tempfile
import os

# ── Config ────────────────────────────────────────────────────
W, H       = 1280, 720
ROBOT_IP   = '192.168.123.164'
MCAST_IP   = '230.1.1.1'
MCAST_PORT = 1720

# ── SDP inline ────────────────────────────────────────────────
sdp = f"""v=0
o=- 0 0 IN IP4 {ROBOT_IP}
s=G1 Camera
c=IN IP4 {MCAST_IP}
t=0 0
a=tool:libavformat
m=video {MCAST_PORT} RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
"""

# Write SDP to temp file (ffmpeg internaly needs a file path for SDP)
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sdp', delete=False)
tmp.write(sdp)
tmp.flush()
tmp.close()

print(f"Connecting to G1 camera at {MCAST_IP}:{MCAST_PORT} ...")

try:
    process = (
        ffmpeg
        .input(
            tmp.name,
            format='sdp',
            fflags='nobuffer',
            flags='low_delay',
            protocol_whitelist='udp,rtp,file,crypto',
            analyzeduration='1000000',
            probesize='1000000'
        )
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    print("Stream opened! Press Q to quit.")

    while True:
        raw = process.stdout.read(W * H * 3)
        if len(raw) != W * H * 3:
            print("Stream ended or error")
            break

        frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
        cv2.imshow("G1 Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    process.terminate()
    cv2.destroyAllWindows()
    os.unlink(tmp.name)
    print("Done")