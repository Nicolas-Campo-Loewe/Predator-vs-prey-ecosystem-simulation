from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from collections import deque

import pygame

CFG = {
    "grid_w": 120,
    "grid_h": 80,
    "fullscreen": False,
    "panel_w": 360,
    "cell_gap": 0,
    "initial_plants": 1800,
    "initial_predators": 120,
    "plant_spread_prob": 0.060,
    "plant_spawn_prob": 0.0015,
    "pred_initial_energy": 22.0,
    "metabolic_cost": 0.20,
    "move_cost": 0.80,
    "eat_gain": 7.0,
    "sense_radius": 6,
    "repro_threshold": 30.0,
    "repro_cost": 10.0,
    "baby_energy": 15.0,
    "three_step_eat": True,
    "gens_per_sec": 12,
    "history_len": 300,
}

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class Predator:
    x: int
    y: int
    energy: float
    eat_wait: int = 0
    eat_target: tuple[int, int] | None = None


def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def neighbors4(x: int, y: int, w: int, h: int):
    for dx, dy in DIRS4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def bfs_find_nearest_plant(
    start: tuple[int, int],
    plants: set[tuple[int, int]],
    w: int,
    h: int,
    radius: int,
) -> tuple[int, int] | None:
    sx, sy = start
    q = deque([(sx, sy, 0)])
    seen = {(sx, sy)}
    while q:
        x, y, d = q.popleft()
        if d > 0 and (x, y) in plants:
            return (x, y)
        if d == radius:
            continue
        for nx, ny in neighbors4(x, y, w, h):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return None


def sign(v: int) -> int:
    return (v > 0) - (v < 0)


def draw_graph(surface: pygame.Surface, rect: pygame.Rect, hist_green, hist_blue, font):
    pygame.draw.rect(surface, (18, 18, 22), rect, border_radius=10)
    pygame.draw.rect(surface, (45, 45, 55), rect, width=2, border_radius=10)

    if len(hist_green) < 2:
        return

    pad = 10
    gx0, gy0 = rect.x + pad, rect.y + pad
    gw = rect.w - 2 * pad
    gh = rect.h - 2 * pad

    max_val = max(max(hist_green), max(hist_blue), 1)
    n = len(hist_green)
    if n < 2:
        return

    def to_xy(i, val):
        x = gx0 + int(gw * (i / (n - 1)))
        y = gy0 + int(gh * (1 - (val / max_val)))
        return x, y

    pts_g = [to_xy(i, v) for i, v in enumerate(hist_green)]
    pts_b = [to_xy(i, v) for i, v in enumerate(hist_blue)]

    pygame.draw.lines(surface, (80, 220, 120), False, pts_g, 2)
    pygame.draw.lines(surface, (90, 160, 255), False, pts_b, 2)

    label_g = font.render("Verdes", True, (220, 220, 230))
    label_b = font.render("Azules", True, (220, 220, 230))
    surface.blit(label_g, (rect.x + 14, rect.y + 10))
    surface.blit(label_b, (rect.x + 120, rect.y + 10))

    pygame.draw.circle(surface, (80, 220, 120), (rect.x + 86, rect.y + 18), 5)
    pygame.draw.circle(surface, (90, 160, 255), (rect.x + 195, rect.y + 18), 5)


def main():
    pygame.init()
    pygame.display.set_caption("Ecosistema (verde vs azul)")

    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    if CFG["fullscreen"]:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(
            (min(1400, screen_w), min(900, screen_h)), pygame.RESIZABLE
        )

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    font_big = pygame.font.SysFont("consolas", 20, bold=True)

    def compute_layout():
        sw, sh = screen.get_size()
        panel_w = min(CFG["panel_w"], max(260, sw // 4))
        sim_w = sw - panel_w

        cell_size = min(sim_w // CFG["grid_w"], sh // CFG["grid_h"])
        cell_size = max(4, cell_size)

        grid_px_w = cell_size * CFG["grid_w"]
        grid_px_h = cell_size * CFG["grid_h"]

        ox = (sim_w - grid_px_w) // 2
        oy = (sh - grid_px_h) // 2

        grid_rect = pygame.Rect(ox, oy, grid_px_w, grid_px_h)
        panel_rect = pygame.Rect(sim_w, 0, panel_w, sh)
        return cell_size, grid_rect, panel_rect

    cell_size, grid_rect, panel_rect = compute_layout()

    w, h = CFG["grid_w"], CFG["grid_h"]
    plants: set[tuple[int, int]] = set()
    predators: list[Predator] = []
    generation = 0

    hist_green = []
    hist_blue = []

    paused = False
    accumulator_s = 0.0

    def reset_world():
        nonlocal plants, predators, generation, hist_green, hist_blue, paused, accumulator_s
        plants = set()
        predators = []
        generation = 0
        hist_green = []
        hist_blue = []
        paused = False
        accumulator_s = 0.0

        while len(plants) < CFG["initial_plants"]:
            plants.add((random.randrange(w), random.randrange(h)))

        occupied = set(plants)
        while len(predators) < CFG["initial_predators"]:
            x, y = random.randrange(w), random.randrange(h)
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            predators.append(Predator(x, y, CFG["pred_initial_energy"]))

    reset_world()

    def cell_at_mouse(mx, my):
        if not grid_rect.collidepoint(mx, my):
            return None
        gx = (mx - grid_rect.x) // cell_size
        gy = (my - grid_rect.y) // cell_size
        if 0 <= gx < w and 0 <= gy < h:
            return int(gx), int(gy)
        return None

    def world_step():
        nonlocal plants, predators, generation, hist_green, hist_blue

        generation += 1

        pred_pos = {(p.x, p.y) for p in predators}

        if plants:
            proposals = []
            for (x, y) in plants:
                if random.random() < CFG["plant_spread_prob"]:
                    neigh = list(neighbors4(x, y, w, h))
                    random.shuffle(neigh)
                    for nx, ny in neigh:
                        if (nx, ny) not in plants and (nx, ny) not in pred_pos:
                            proposals.append((nx, ny))
                            break
            random.shuffle(proposals)
            for nx, ny in proposals:
                if (nx, ny) not in plants and (nx, ny) not in pred_pos:
                    plants.add((nx, ny))

        if CFG["plant_spawn_prob"] > 0:
            trials = int(w * h * CFG["plant_spawn_prob"])
            for _ in range(trials):
                x, y = random.randrange(w), random.randrange(h)
                if (x, y) not in plants and (x, y) not in pred_pos:
                    if random.random() < 0.65:
                        plants.add((x, y))

        new_predators: list[Predator] = []
        pred_pos = {(p.x, p.y) for p in predators}

        plants_set = plants
        for p in predators:
            p.energy -= CFG["metabolic_cost"]

            if CFG["three_step_eat"] and p.eat_wait > 0:
                p.eat_wait -= 1
                if p.eat_wait == 0 and p.eat_target is not None:
                    tx, ty = p.eat_target
                    if (tx, ty) in plants_set and (
                        abs(tx - p.x) + abs(ty - p.y) == 1
                    ):
                        if (tx, ty) not in pred_pos:
                            pred_pos.remove((p.x, p.y))
                            p.x, p.y = tx, ty
                            pred_pos.add((p.x, p.y))

                        if (tx, ty) in plants_set:
                            plants_set.remove((tx, ty))
                            p.energy += CFG["eat_gain"]

                    p.eat_target = None

        pred_pos = {(p.x, p.y) for p in predators}

        move_proposals = {}
        for idx, p in enumerate(predators):
            if p.energy <= 0:
                continue
            if CFG["three_step_eat"] and p.eat_wait > 0:
                continue

            target = bfs_find_nearest_plant(
                (p.x, p.y), plants_set, w, h, CFG["sense_radius"]
            )

            def propose(nx, ny):
                if (nx, ny) in plants_set:
                    return False
                if (nx, ny) in pred_pos:
                    return False
                move_proposals.setdefault((nx, ny), []).append(idx)
                return True

            moved = False
            if target is not None:
                tx, ty = target
                dx = sign(tx - p.x)
                dy = sign(ty - p.y)
                candidates = []
                if dx != 0:
                    candidates.append((p.x + dx, p.y))
                if dy != 0:
                    candidates.append((p.x, p.y + dy))
                random.shuffle(candidates)
                for nx, ny in candidates:
                    if in_bounds(nx, ny, w, h) and propose(nx, ny):
                        moved = True
                        break

            if not moved:
                neigh = list(neighbors4(p.x, p.y, w, h))
                random.shuffle(neigh)
                for nx, ny in neigh:
                    if propose(nx, ny):
                        moved = True
                        break

        winners = {}
        for dest, idxs in move_proposals.items():
            if len(idxs) == 1:
                winners[idxs[0]] = dest
            else:
                winners[random.choice(idxs)] = dest

        pred_pos = {(p.x, p.y) for p in predators}
        for idx, dest in winners.items():
            p = predators[idx]
            if p.energy <= 0:
                continue
            nx, ny = dest
            if (nx, ny) in pred_pos:
                continue
            pred_pos.remove((p.x, p.y))
            p.x, p.y = nx, ny
            pred_pos.add((p.x, p.y))
            p.energy -= CFG["move_cost"]

        if CFG["three_step_eat"]:
            pred_pos = {(p.x, p.y) for p in predators}
            order = list(range(len(predators)))
            random.shuffle(order)
            for idx in order:
                p = predators[idx]
                if p.energy <= 0 or p.eat_wait > 0:
                    continue
                adj_plants = []
                for nx, ny in neighbors4(p.x, p.y, w, h):
                    if (nx, ny) in plants_set:
                        adj_plants.append((nx, ny))
                if adj_plants:
                    p.eat_target = random.choice(adj_plants)
                    p.eat_wait = 2

        pred_pos = {(p.x, p.y) for p in predators}
        for p in predators:
            if p.energy <= 0:
                continue
            if p.energy >= CFG["repro_threshold"]:
                neigh = list(neighbors4(p.x, p.y, w, h))
                random.shuffle(neigh)
                for nx, ny in neigh:
                    if (nx, ny) not in plants_set and (nx, ny) not in pred_pos:
                        p.energy -= CFG["repro_cost"]
                        baby = Predator(nx, ny, CFG["baby_energy"])
                        new_predators.append(baby)
                        pred_pos.add((nx, ny))
                        break

        predators = [p for p in predators if p.energy > 0]
        predators.extend(new_predators)

        hist_green.append(len(plants_set))
        hist_blue.append(len(predators))
        if len(hist_green) > CFG["history_len"]:
            hist_green = hist_green[-CFG["history_len"] :]
            hist_blue = hist_blue[-CFG["history_len"] :]

        plants = plants_set

    def clamp_cfg(key, lo, hi):
        CFG[key] = max(lo, min(hi, CFG[key]))

    running = True
    while running:
        dt_s = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE and not CFG["fullscreen"]:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                cell_size, grid_rect, panel_rect = compute_layout()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    accumulator_s = 0.0
                elif event.key == pygame.K_n:
                    if paused:
                        world_step()
                elif event.key == pygame.K_x:
                    reset_world()
                elif event.key == pygame.K_f:
                    CFG["fullscreen"] = not CFG["fullscreen"]
                    if CFG["fullscreen"]:
                        screen = pygame.display.set_mode(
                            (info.current_w, info.current_h), pygame.FULLSCREEN
                        )
                    else:
                        screen = pygame.display.set_mode(
                            (min(1400, info.current_w), min(900, info.current_h)),
                            pygame.RESIZABLE,
                        )
                    cell_size, grid_rect, panel_rect = compute_layout()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                cell = cell_at_mouse(mx, my)
                if cell:
                    gx, gy = cell
                    mods = pygame.key.get_mods()
                    shift = bool(mods & pygame.KMOD_SHIFT)
                    if shift:
                        plants.discard((gx, gy))
                        for i in range(len(predators) - 1, -1, -1):
                            if predators[i].x == gx and predators[i].y == gy:
                                predators.pop(i)
                    else:
                        if event.button == 1:
                            if not any(p.x == gx and p.y == gy for p in predators):
                                plants.add((gx, gy))
                        elif event.button == 3:
                            if (gx, gy) not in plants and not any(
                                p.x == gx and p.y == gy for p in predators
                            ):
                                predators.append(Predator(gx, gy, CFG["pred_initial_energy"]))

        keys = pygame.key.get_pressed()

        if keys[pygame.K_g]:
            CFG["plant_spread_prob"] += 0.002
        if keys[pygame.K_h]:
            CFG["plant_spread_prob"] -= 0.002

        if keys[pygame.K_e]:
            CFG["plant_spawn_prob"] += 0.0002
        if keys[pygame.K_d]:
            CFG["plant_spawn_prob"] -= 0.0002

        if keys[pygame.K_v]:
            CFG["sense_radius"] += 1
        if keys[pygame.K_b]:
            CFG["sense_radius"] -= 1
        if keys[pygame.K_t]:
            CFG["eat_gain"] += 0.2
        if keys[pygame.K_y]:
            CFG["eat_gain"] -= 0.2
        if keys[pygame.K_u]:
            CFG["move_cost"] += 0.05
        if keys[pygame.K_j]:
            CFG["move_cost"] -= 0.05
        if keys[pygame.K_i]:
            CFG["repro_threshold"] += 0.5
        if keys[pygame.K_k]:
            CFG["repro_threshold"] -= 0.5
        if keys[pygame.K_o]:
            CFG["repro_cost"] += 0.5
        if keys[pygame.K_l]:
            CFG["repro_cost"] -= 0.5

        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            CFG["gens_per_sec"] += 1
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            CFG["gens_per_sec"] -= 1

        clamp_cfg("plant_spread_prob", 0.0, 0.30)
        clamp_cfg("plant_spawn_prob", 0.0, 0.02)
        CFG["sense_radius"] = max(1, min(20, int(CFG["sense_radius"])))
        clamp_cfg("eat_gain", 0.0, 50.0)
        clamp_cfg("move_cost", 0.0, 10.0)
        clamp_cfg("repro_threshold", 1.0, 200.0)
        clamp_cfg("repro_cost", 0.0, 200.0)
        CFG["gens_per_sec"] = max(1, min(240, int(CFG["gens_per_sec"])))

        if not paused:
            accumulator_s += dt_s
            step_interval = 1.0 / CFG["gens_per_sec"]

            max_steps_per_frame = 200
            steps = 0
            while accumulator_s >= step_interval and steps < max_steps_per_frame:
                world_step()
                accumulator_s -= step_interval
                steps += 1

        screen.fill((10, 10, 12))

        pygame.draw.rect(screen, (14, 14, 18), grid_rect)

        gap = CFG["cell_gap"]
        for (x, y) in plants:
            rx = grid_rect.x + x * cell_size + gap
            ry = grid_rect.y + y * cell_size + gap
            pygame.draw.rect(
                screen,
                (70, 220, 120),
                pygame.Rect(rx, ry, cell_size - 2 * gap, cell_size - 2 * gap),
            )

        for p in predators:
            rx = grid_rect.x + p.x * cell_size + gap
            ry = grid_rect.y + p.y * cell_size + gap
            pygame.draw.rect(
                screen,
                (90, 160, 255),
                pygame.Rect(rx, ry, cell_size - 2 * gap, cell_size - 2 * gap),
            )

        pygame.draw.rect(screen, (12, 12, 15), panel_rect)

        graph_rect = pygame.Rect(
            panel_rect.x + 14,
            110,
            panel_rect.w - 28,
            min(260, panel_rect.h - 220),
        )
        draw_graph(screen, graph_rect, hist_green, hist_blue, font)

        title = font_big.render("ECOSISTEMA", True, (235, 235, 245))
        screen.blit(title, (panel_rect.x + 14, 14))

        status = "PAUSA" if paused else "RUN"
        status_s = font_big.render(status, True, (235, 235, 245))
        screen.blit(status_s, (panel_rect.x + 14, 40))

        lines = [
            f"Gen: {generation}",
            f"Verdes: {len(plants)}",
            f"Azules: {len(predators)}",
            "",
            f"plant_spread_prob: {CFG['plant_spread_prob']:.3f}  (G/H)",
            f"plant_spawn_prob : {CFG['plant_spawn_prob']:.4f}  (E/D)",
            f"sense_radius     : {CFG['sense_radius']}      (V/B)",
            f"eat_gain         : {CFG['eat_gain']:.1f}    (T/Y)",
            f"move_cost        : {CFG['move_cost']:.2f}   (U/J)",
            f"repro_threshold  : {CFG['repro_threshold']:.1f} (I/K)",
            f"repro_cost       : {CFG['repro_cost']:.1f}  (O/L)",
            f"generations/sec  : {CFG['gens_per_sec']}      (+/-)",
            "",
            "SPACE pause | N step | X reset",
            "Mouse: L=verde R=azul | Shift=borra",
        ]

        y = graph_rect.bottom + 14
        for s in lines:
            txt = font.render(s, True, (210, 210, 220))
            screen.blit(txt, (panel_rect.x + 14, y))
            y += 18

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
