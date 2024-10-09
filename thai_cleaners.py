import re

def thai_cleaners(text):
    # ลบช่องว่างซ้ำ
    text = re.sub(r'\s+', ' ', text)
    
    # ลบช่องว่างหน้าและหลังข้อความ
    text = text.strip()
    
    # แปลงตัวเลขเป็นคำอ่าน (ตัวอย่างง่ายๆ สำหรับเลข 0-10)
    number_map = {
        '0': 'ศูนย์', '1': 'หนึ่ง', '2': 'สอง', '3': 'สาม', '4': 'สี่', '5': 'ห้า',
        '6': 'หก', '7': 'เจ็ด', '8': 'แปด', '9': 'เก้า', '10': 'สิบ'
    }
    for num, word in number_map.items():
        text = text.replace(num, word)
    
    # เพิ่มกฎการทำความสะอาดอื่นๆ ตามต้องการ
    
    return text