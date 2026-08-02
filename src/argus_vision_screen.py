import dxcam
import cv2
import time

class ArgusEyes:
    def __init__(self, target_fps=30):
        print("[SYSTEM] Initializing A.R.G.U.S. Optical Sensors (DXcam)...")
        # DXcam is highly optimized for Windows. 
        # output_color="BGR" ensures it plays nicely with OpenCV.
        self.camera = dxcam.create(output_color="BGR")
        
        # Start a background thread that constantly captures the screen
        self.camera.start(target_fps=target_fps)
        print(f"[SYSTEM] Optical sensors active at {target_fps} FPS.")

    def get_latest_frame(self):
        """
        Grabs the absolute newest frame from the buffer.
        If the screen hasn't changed, it safely returns None.
        """
        return self.camera.get_latest_frame()

    def shutdown(self):
        print("[SYSTEM] Shutting down optical sensors...")
        self.camera.stop()

# ==========================================
# TEST ENGINE
# ==========================================
if __name__ == "__main__":
    eyes = ArgusEyes(target_fps=30)
    print("\n[SYSTEM] A.R.G.U.S. is now watching your screen.")
    print("[SYSTEM] Bring up any video or application. Press 'q' in the preview window to quit.")
    
    try:
        while True:
            # Grab the frame in real-time
            frame = eyes.get_latest_frame()
            
            if frame is not None:
                # We resize the preview so it doesn't take up your entire monitor,
                # but the actual AI will see the full-resolution version!
                preview = cv2.resize(frame, (960, 540))
                cv2.imshow("A.R.G.U.S. Vision Preview", preview)
            
            # Listen for the 'q' key to shut down gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        eyes.shutdown()
        cv2.destroyAllWindows()