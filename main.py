import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import random
import time

# ══════════════════════════════════════════════════════════════
# 1. KONFIGURASI GLOBAL & DATABASE
# ══════════════════════════════════════════════════════════════

CAM_W, CAM_H = 1280, 720
SNAP_DIST = 140

PINCH_FINGER_A = 8
PINCH_FINGER_B = 12
PINCH_DISTANCE = 45

COLOR_MAP = {
    "H":  (255, 255, 100),
    "O":  (100, 100, 255),
    "C":  (100, 100, 100),
    "Na": (50,  200, 255),
    "Cl": (100, 255, 100),
    "N":  (255, 100, 100),
    "Fe": (50,  100, 200),
    "Cu": (50,  150, 200),
    "S":  (50,  255, 200)
}

NAME_MAP = {
    "H": "Hidrogen",
    "O": "Oksigen",
    "C": "Karbon",
    "Na": "Natrium",
    "Cl": "Klorin",
    "N": "Nitrogen",
    "Fe": "Besi",
    "Cu": "Tembaga",
    "S": "Sulfur",

    "H2": "Gas Hidrogen",
    "O2": "Gas Oksigen",
    "N2": "Gas Nitrogen",
    "OH": "Hidroksil",
    "H2O": "Air",
    "CO": "Karbon Monoksida",
    "CO2": "Karbon Dioksida",
    "CH4": "Metana",
    "NaCl": "Garam Dapur",
    "HCl": "Asam Klorida",
    "NaOH": "Soda Api",
    "FeO": "Besi Oksida",
    "CuS": "Tembaga Sulfida",
    "SO2": "Sulfur Dioksida"
}

RECIPES = {
    tuple(sorted(["H", "H"])): {
        "result": "H2",
        "color": (255, 255, 200),
        "desc": "Gas Hidrogen: Sangat ringan dan mudah terbakar."
    },
    tuple(sorted(["O", "O"])): {
        "result": "O2",
        "color": (200, 200, 255),
        "desc": "Gas Oksigen: Dibutuhkan untuk bernapas dan api."
    },
    tuple(sorted(["N", "N"])): {
        "result": "N2",
        "color": (255, 150, 150),
        "desc": "Gas Nitrogen: 78 persen penyusun atmosfer Bumi."
    },

    tuple(sorted(["H", "O"])): {
        "result": "OH",
        "color": (200, 150, 200),
        "desc": "Hidroksil: Butuh 1 H lagi untuk menjadi air."
    },
    tuple(sorted(["OH", "H"])): {
        "result": "H2O",
        "color": (255, 150, 50),
        "desc": "Air (H2O): Basis dari segala kehidupan."
    },

    tuple(sorted(["C", "O"])): {
        "result": "CO",
        "color": (150, 150, 200),
        "desc": "Karbon Monoksida: Beracun, tambah O lagi."
    },
    tuple(sorted(["CO", "O"])): {
        "result": "CO2",
        "color": (150, 255, 150),
        "desc": "Karbon Dioksida: Gas hasil pernapasan."
    },
    tuple(sorted(["C", "H"])): {
        "result": "CH4",
        "color": (50, 200, 200),
        "desc": "Metana: Komponen utama gas alam."
    },

    tuple(sorted(["Na", "Cl"])): {
        "result": "NaCl",
        "color": (0, 215, 255),
        "desc": "Garam Dapur: Natrium Klorida."
    },
    tuple(sorted(["H", "Cl"])): {
        "result": "HCl",
        "color": (50, 255, 50),
        "desc": "Asam Klorida: Asam kuat."
    },
    tuple(sorted(["Na", "OH"])): {
        "result": "NaOH",
        "color": (200, 50, 200),
        "desc": "Soda Api: Basa kuat."
    },

    tuple(sorted(["Fe", "O"])): {
        "result": "FeO",
        "color": (50, 50, 200),
        "desc": "Besi Oksida: Awal terbentuknya karat."
    },
    tuple(sorted(["Cu", "S"])): {
        "result": "CuS",
        "color": (50, 100, 100),
        "desc": "Tembaga Sulfida: Senyawa mineral."
    },
    tuple(sorted(["S", "O"])): {
        "result": "SO2",
        "color": (100, 200, 150),
        "desc": "Sulfur Dioksida: Penyebab hujan asam."
    }
}

LEVELS = {
    1: {
        "name": "Level 1: Eksperimen Air",
        "target": "H2O",
        "hint": "Gabungkan H + O menjadi OH, lalu OH + H menjadi H2O.",
        "elements": ["H", "H", "O", "O"]
    },
    2: {
        "name": "Level 2: Gas Karbon",
        "target": "CO2",
        "hint": "Gabungkan C + O menjadi CO, lalu CO + O menjadi CO2.",
        "elements": ["C", "O", "O", "H"]
    },
    3: {
        "name": "Level 3: Dapur Kimia",
        "target": "NaCl",
        "hint": "Gabungkan Na dan Cl untuk membuat garam dapur.",
        "elements": ["Na", "Cl", "H", "O"]
    },
    4: {
        "name": "Level 4: Mineral dan Logam",
        "target": "CuS",
        "hint": "Coba gabungkan logam dan mineral yang tersedia.",
        "elements": ["Fe", "Cu", "O", "S"]
    },
    5: {
        "name": "Level 5: Eksperimen Bebas",
        "target": "Bebas",
        "hint": "Temukan kombinasi reaksi sebanyak mungkin.",
        "elements": ["Na", "Cl", "C", "O", "O", "H"]
    }
}


# ══════════════════════════════════════════════════════════════
# 2. FULL HAND TRACKING
# ══════════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (5, 17)
]


def draw_corner_box(img, x, y, w, h, color=(0, 255, 0), length=25, thickness=4):
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)

    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)

    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)

    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)


def draw_full_hand(img, hand, is_pinching=False):
    lmList = hand["lmList"]

    line_color = (0, 255, 0) if is_pinching else (0, 180, 255)
    point_color = (255, 0, 255)

    for a, b in HAND_CONNECTIONS:
        x1, y1 = lmList[a][:2]
        x2, y2 = lmList[b][:2]
        cv2.line(img, (x1, y1), (x2, y2), line_color, 2)

    for i, lm in enumerate(lmList):
        x, y = lm[:2]
        radius = 7 if i in [4, 8, 12, 16, 20] else 4
        cv2.circle(img, (x, y), radius, point_color, -1)

    if "bbox" in hand:
        x, y, w, h = hand["bbox"]
        draw_corner_box(img, x, y, w, h, line_color)

    for idx in [PINCH_FINGER_A, PINCH_FINGER_B]:
        x, y = lmList[idx][:2]
        cv2.circle(img, (x, y), 13, line_color, 2)


# ══════════════════════════════════════════════════════════════
# 3. CLASS VISUAL
# ══════════════════════════════════════════════════════════════

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(5, 25)

        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.color = color
        self.radius = random.uniform(5, 15)
        self.alpha = 1.0

    def update_and_draw(self, img):
        if self.alpha <= 0:
            return

        self.x += self.vx
        self.y += self.vy
        self.vy += 1.2

        self.radius = max(0, self.radius - 0.4)
        self.alpha -= 0.04

        if self.radius > 0:
            overlay = img.copy()
            cv2.circle(
                overlay,
                (int(self.x), int(self.y)),
                int(self.radius),
                self.color,
                -1
            )
            cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0, img)


def draw_text_fit_center(img, text, center_x, y, max_width,
                         font, max_scale, min_scale, color, thickness):
    scale = max_scale

    while scale >= min_scale:
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)

        if text_size[0] <= max_width:
            break

        scale -= 0.02

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center_x - text_size[0] // 2)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness
    )


class ElementBox:
    def __init__(self, cx, cy, label, color=None, desc=None):
        self.cx = cx
        self.cy = cy
        self.w = 120
        self.h = 120

        self.label = label
        self.name = NAME_MAP.get(label, label)
        self.desc = desc

        self.grabbed = False
        self.offset_x = 0
        self.offset_y = 0

        self.created_time = time.time()
        self.color = color if color else COLOR_MAP.get(label, (150, 150, 150))

    def is_hovered(self, x, y):
        return (
            self.cx - self.w // 2 < x < self.cx + self.w // 2 and
            self.cy - self.h // 2 < y < self.cy + self.h // 2
        )

    def draw(self, img):
        # Ukuran kotak tetap, tanpa efek denyut
        w = self.w
        h = self.h

        x1 = int(self.cx - w // 2)
        y1 = int(self.cy - h // 2)
        x2 = int(self.cx + w // 2)
        y2 = int(self.cy + h // 2)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, -1)

        cv2.addWeighted(
            overlay,
            0.75 if self.desc else 0.52,
            img,
            0.25 if self.desc else 0.48,
            0,
            img
        )

        border_color = (0, 255, 0) if self.grabbed else (255, 255, 255)
        thickness = 5 if self.grabbed else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness)

        # Simbol unsur/molekul
        ts, _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_DUPLEX, 1.25, 3)

        cv2.putText(
            img,
            self.label,
            (int(self.cx - ts[0] // 2), int(self.cy - 5)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.25,
            (255, 255, 255),
            3
        )

        # Nama lengkap unsur/molekul dibuat otomatis mengecil agar tidak terpotong
        draw_text_fit_center(
            img,
            self.name,
            self.cx,
            int(self.cy + 30),
            self.w - 14,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            0.25,
            (255, 255, 255),
            1
        )

        # Deskripsi hanya muncul pada hasil reaksi
        if self.desc:
            short_desc = self.desc[:52]
            d_ts, _ = cv2.getTextSize(short_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            dx1 = int(self.cx - d_ts[0] // 2 - 10)
            dy1 = int(self.cy + h // 2 + 10)
            dx2 = int(self.cx + d_ts[0] // 2 + 10)
            dy2 = int(dy1 + d_ts[1] + 16)

            dx1 = max(10, dx1)
            dx2 = min(CAM_W - 10, dx2)

            overlay_text = img.copy()
            cv2.rectangle(overlay_text, (dx1, dy1), (dx2, dy2), (10, 10, 10), -1)
            cv2.addWeighted(overlay_text, 0.8, img, 0.2, 0, img)
            cv2.rectangle(img, (dx1, dy1), (dx2, dy2), self.color, 1)

            cv2.putText(
                img,
                short_desc,
                (dx1 + 10, dy2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


# ══════════════════════════════════════════════════════════════
# 4. FUNGSI GAME
# ══════════════════════════════════════════════════════════════

def load_level(level_id):
    boxes = []
    lvl_data = LEVELS[level_id]
    elements = lvl_data["elements"]

    spacing_x = CAM_W // (len(elements) + 1)

    for i, label in enumerate(elements):
        x = spacing_x * (i + 1)
        y = (CAM_H // 2) + (-40 if i % 2 == 0 else 40)
        boxes.append(ElementBox(x, y, label))

    return boxes, lvl_data["name"]


def wrap_text_by_pixel(text, max_width, font, font_scale, thickness):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = word if current_line == "" else current_line + " " + word
        text_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_size[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def draw_guide_panel(img, level_id):
    level_data = LEVELS[level_id]

    x, y, w = 760, 35, 480
    padding_x = 18

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1

    hint_lines = wrap_text_by_pixel(
        "Tips: " + level_data["hint"],
        w - (padding_x * 2),
        font,
        font_scale,
        thickness
    )

    lines = [
        level_data["name"],
        f"Target: {level_data['target']}"
    ] + hint_lines

    h = 58 + (len(lines) * 24) + 12

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.putText(
        img,
        "PANDUAN",
        (x + padding_x, y + 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    yy = y + 62

    for line in lines:
        cv2.putText(
            img,
            line,
            (x + padding_x, yy),
            font,
            font_scale,
            (235, 235, 235),
            thickness
        )

        yy += 24


# ══════════════════════════════════════════════════════════════
# 5. LOOP UTAMA
# ══════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    current_level = 1
    boxes, level_name = load_level(current_level)
    particles = []

    print("\n[INFO] Chemistry Sandbox Lab Berjalan!")
    print("[INFO] Tekan 1-5 untuk ganti level, R untuk reset, Q untuk keluar.")

    while True:
        success, img = cap.read()

        if not success:
            break

        img = cv2.flip(img, 1)

        # Background gelap transparan
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (CAM_W, CAM_H), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        hands, img = detector.findHands(img, draw=False)

        cursor_x, cursor_y = 0, 0
        is_pinching = False
        active_hand = None

        if hands:
            active_hand = hands[0]
            lmList = active_hand["lmList"]

            cursor_x, cursor_y = lmList[8][:2]

            x1, y1 = lmList[PINCH_FINGER_A][:2]
            x2, y2 = lmList[PINCH_FINGER_B][:2]

            dist = math.hypot(x1 - x2, y1 - y2)
            is_pinching = dist < PINCH_DISTANCE

        # Drag & drop
        for box in reversed(boxes):
            if is_pinching:
                if (
                    not box.grabbed and
                    box.is_hovered(cursor_x, cursor_y) and
                    not any(b.grabbed for b in boxes)
                ):
                    box.grabbed = True
                    box.offset_x = cursor_x - box.cx
                    box.offset_y = cursor_y - box.cy

                if box.grabbed:
                    box.cx = cursor_x - box.offset_x
                    box.cy = cursor_y - box.offset_y

                    box.cx = max(65, min(CAM_W - 65, box.cx))
                    box.cy = max(150, min(CAM_H - 65, box.cy))
            else:
                box.grabbed = False

        # Proses reaksi
        to_remove = []
        to_add = []

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1 = boxes[i]
                b2 = boxes[j]

                if b1 in to_remove or b2 in to_remove:
                    continue

                distance = math.hypot(b1.cx - b2.cx, b1.cy - b2.cy)

                if distance < SNAP_DIST and not b1.grabbed and not b2.grabbed:
                    pair_key = tuple(sorted([b1.label, b2.label]))

                    if pair_key in RECIPES:
                        recipe = RECIPES[pair_key]

                        mid_x = (b1.cx + b2.cx) // 2
                        mid_y = (b1.cy + b2.cy) // 2

                        to_remove.extend([b1, b2])

                        to_add.append(
                            ElementBox(
                                mid_x,
                                mid_y,
                                recipe["result"],
                                recipe["color"],
                                recipe["desc"]
                            )
                        )

                        for _ in range(40):
                            particles.append(Particle(mid_x, mid_y, recipe["color"]))

        for box in to_remove:
            if box in boxes:
                boxes.remove(box)

        boxes.extend(to_add)

        # Gambar objek
        for box in boxes:
            if not box.grabbed:
                box.draw(img)

        for box in boxes:
            if box.grabbed:
                box.draw(img)

        # Gambar partikel
        for p in particles:
            p.update_and_draw(img)

        particles = [p for p in particles if p.alpha > 0]

        # Gambar tangan penuh
        if active_hand:
            draw_full_hand(img, active_hand, is_pinching)

            cursor_color = (0, 255, 0) if is_pinching else (255, 255, 255)

            cv2.circle(
                img,
                (cursor_x, cursor_y),
                15,
                cursor_color,
                -1 if is_pinching else 2
            )

            mode_text = "DRAG" if is_pinching else "OPEN"

            cv2.putText(
                img,
                mode_text,
                (cursor_x + 20, cursor_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cursor_color,
                2
            )

        # Header utama
        cv2.putText(
            img,
            "Chemistry Sandbox Lab",
            (30, 55),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (255, 230, 100),
            2
        )

        instruction_text = "Level: [1] [2] [3] [4] [5]   |   Reset: [R]   |   Keluar: [Q]"

        cv2.putText(
            img,
            instruction_text,
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (210, 210, 210),
            1
        )

        # Panel panduan
        draw_guide_panel(img, current_level)

        cv2.imshow("Chemistry Sandbox Lab", img)

        key = cv2.waitKey(1)

        if key in [ord('q'), 27]:
            break

        elif key == ord('r'):
            boxes, level_name = load_level(current_level)
            particles.clear()

        elif ord('1') <= key <= ord('5'):
            current_level = int(chr(key))
            boxes, level_name = load_level(current_level)
            particles.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()