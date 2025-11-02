# โค้ดนี้ทำงานได้เฉพาะบน Windows เท่านั้น

def find_camera_names_windows():
    """
    ฟังก์ชันสำหรับดึงรายชื่อกล้องวิดีโอทั้งหมดบน Windows
    """
    try:
        # จำเป็นต้อง import ภายใน try-except เพราะ library นี้มีเฉพาะบน Windows
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        devices = graph.get_input_devices() # ดึงรายชื่ออุปกรณ์วิดีโอ
        return devices
    except ImportError:
        print("ไม่สามารถ import 'pygrabber' ได้")
        print("Library นี้จำเป็นสำหรับการทำงานบน Windows")
        print("กรุณาติดตั้งด้วยคำสั่ง: pip install pygrabber")
        return []
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        return []


if __name__ == "__main__":
    print("กำลังค้นหาชื่อกล้อง (สำหรับ Windows เท่านั้น)...")
    camera_names = find_camera_names_windows()

    if camera_names:
        print("\nพบกล้องทั้งหมด:")
        # แสดงผลลัพธ์เป็น Index คู่กับชื่อ
        for index, name in enumerate(camera_names):
            print(f"  Index {index}: {name}")
    else:
        print("\nไม่พบกล้อง หรืออาจจะไม่ได้ใช้ระบบปฏิบัติการ Windows")