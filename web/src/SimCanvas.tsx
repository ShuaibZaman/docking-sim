import { useEffect, useRef, type MouseEvent } from "react";
import type { Frame, Selection, World } from "./types";

type Props = {
  world: World | null;
  frames: Frame[];
  index: number;
  ghost?: Frame[] | null;
  selection: Selection | null;
  onSelect: (selection: Selection | null) => void;
};

type Rect = { x: number; y: number; w: number; h: number };
type Box = { cx: number; cy: number; w: number; h: number };

function wx(x: number, world: World, width: number): number {
  return ((x - world.x_min) / (world.x_max - world.x_min)) * width;
}

function wy(y: number, world: World, height: number): number {
  return (1 - (y - world.y_min) / (world.y_max - world.y_min)) * height;
}

function shipRadiusPx(world: World, width: number): number {
  return Math.max(6, (world.ship_radius / (world.x_max - world.x_min)) * width * 1.8);
}

function boxRect(box: Box, world: World, width: number, height: number): Rect {
  return {
    x: wx(box.cx - box.w / 2, world, width),
    y: wy(box.cy + box.h / 2, world, height),
    w: (box.w / (world.x_max - world.x_min)) * width,
    h: (box.h / (world.y_max - world.y_min)) * height,
  };
}

function pointInRect(px: number, py: number, rect: Rect): boolean {
  return px >= rect.x && px <= rect.x + rect.w && py >= rect.y && py <= rect.y + rect.h;
}

export function hitTest(
  world: World,
  frames: Frame[],
  index: number,
  width: number,
  height: number,
  px: number,
  py: number,
): Selection | null {
  const frame = frames[index];
  if (!frame || width <= 0 || height <= 0) return null;

  const radius = shipRadiusPx(world, width);
  const sx = wx(frame.x, world, width);
  const sy = wy(frame.y, world, height);
  if ((px - sx) ** 2 + (py - sy) ** 2 <= (radius * 1.45) ** 2) {
    return { kind: "ship" };
  }
  if (pointInRect(px, py, boxRect(world.port, world, width, height))) {
    return { kind: "port" };
  }
  if (pointInRect(px, py, boxRect(world.hull, world, width, height))) {
    return { kind: "station" };
  }

  const start = Math.max(0, index - 80);
  let best: { index: number; dist: number } | null = null;
  for (let i = start; i <= index && i < frames.length; i += 1) {
    const point = frames[i];
    const dist = Math.hypot(px - wx(point.x, world, width), py - wy(point.y, world, height));
    if (dist <= 10 && (best === null || dist < best.dist)) {
      best = { index: i, dist };
    }
  }
  return best ? { kind: "trail", index: best.index } : null;
}

function drawStars(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.fillStyle = "#07090f";
  ctx.fillRect(0, 0, width, height);
  for (let i = 0; i < 90; i += 1) {
    const x = (((i * 127 + 19) % 1000) / 1000) * width;
    const y = (((i * 311 + 53) % 1000) / 1000) * height;
    const a = 0.25 + ((i * 17) % 50) / 100;
    ctx.fillStyle = `rgba(220, 228, 240, ${a})`;
    ctx.fillRect(x, y, i % 7 === 0 ? 2 : 1, i % 7 === 0 ? 2 : 1);
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
) {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const head = 7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(angle - 0.45), y1 - head * Math.sin(angle - 0.45));
  ctx.lineTo(x1 - head * Math.cos(angle + 0.45), y1 - head * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
}

function drawPlume(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  angle: number,
  strength: number,
  radius: number,
) {
  if (strength < 0.05) return;
  const mag = Math.min(1, strength);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = `rgba(255, 140, 70, ${0.35 + mag * 0.5})`;
  ctx.beginPath();
  ctx.moveTo(-radius * 0.2, 0);
  ctx.lineTo(-radius * (1.6 + mag), radius * 0.45);
  ctx.lineTo(-radius * (1.6 + mag), -radius * 0.45);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLegend(ctx: CanvasRenderingContext2D) {
  const items: Array<[string, string, "box" | "line"]> = [
    ["#f3f1ea", "Ship", "box"],
    ["#3ee0c5", "Port", "box"],
    ["#8b93a3", "Hull", "box"],
    ["#e8a54b", "Path", "line"],
    ["#7eb6ff", "Velocity", "line"],
  ];
  ctx.fillStyle = "rgba(7, 9, 15, 0.78)";
  ctx.fillRect(12, 12, 108, items.length * 16 + 10);
  ctx.font = "11px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  items.forEach(([color, label, mark], i) => {
    const y = 28 + i * 16;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    if (mark === "box") {
      ctx.fillRect(22, y - 4, 10, 8);
    } else {
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(20, y);
      ctx.lineTo(34, y);
      ctx.stroke();
    }
    ctx.fillStyle = "#d6dde8";
    ctx.fillText(label, 40, y);
  });
}

function roundBox(
  ctx: CanvasRenderingContext2D,
  rect: Rect,
  radius: number,
  fill: string,
  stroke: string,
  lineWidth: number,
) {
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.roundRect(rect.x, rect.y, rect.w, rect.h, radius);
  ctx.fill();
  ctx.stroke();
}

export function SimCanvas({ world, frames, index, ghost, selection, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef({ world, frames, index });
  stateRef.current = { world, frames, index };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawStars(ctx, width, height);
      if (!world) return;

      const hull = boxRect(world.hull, world, width, height);
      const hullSelected = selection?.kind === "station";
      roundBox(ctx, hull, 6, "#2a3344", hullSelected ? "#f3f1ea" : "#8b93a3", hullSelected ? 2.5 : 1.5);
      ctx.fillStyle = "#d8deea";
      ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      const hullCenterX = wx(world.hull.cx, world, width);
      ctx.fillText("STATION", hullCenterX, hull.y + hull.h * 0.42);
      ctx.fillStyle = "#e85d4c";
      ctx.font = "10px 'IBM Plex Sans', sans-serif";
      ctx.fillText("collision", hullCenterX, hull.y + hull.h * 0.72);

      const port = boxRect(world.port, world, width, height);
      const portSelected = selection?.kind === "port";
      roundBox(
        ctx,
        port,
        3,
        "rgba(62, 224, 197, 0.18)",
        portSelected ? "#f3f1ea" : "#3ee0c5",
        portSelected ? 2.5 : 1.5,
      );
      const speedLimit = world.dock_speed_max ?? 0.35;
      const angleLimit = world.dock_angle_max_deg ?? 12;
      ctx.fillStyle = "#3ee0c5";
      ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
      ctx.fillText("PORT", wx(world.port.cx, world, width), port.y + port.h + 14);
      ctx.font = "10px 'IBM Plex Sans', sans-serif";
      ctx.fillStyle = "#8b93a3";
      ctx.fillText(
        `speed \u2264 ${speedLimit}  \u00b7  \u00b1${Math.round(angleLimit)}\u00b0`,
        wx(world.port.cx, world, width),
        port.y + port.h + 28,
      );

      if (ghost && ghost.length > 1) {
        const ghostEnd = Math.min(index, ghost.length - 1);
        const ghostTrail = ghost.slice(0, ghostEnd + 1);
        ctx.beginPath();
        ghostTrail.forEach((frame, i) => {
          const x = wx(frame.x, world, width);
          const y = wy(frame.y, world, height);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = "rgba(126, 182, 255, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const trail = frames.slice(Math.max(0, index - 80), index + 1);
      if (trail.length > 1) {
        ctx.beginPath();
        trail.forEach((frame, i) => {
          const x = wx(frame.x, world, width);
          const y = wy(frame.y, world, height);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = "rgba(232, 165, 75, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (selection?.kind === "trail") {
        const picked = frames[selection.index];
        if (picked) {
          ctx.fillStyle = "#f3f1ea";
          ctx.beginPath();
          ctx.arc(wx(picked.x, world, width), wy(picked.y, world, height), 4.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const frame = frames[index];
      if (!frame) {
        drawLegend(ctx);
        return;
      }
      const x = wx(frame.x, world, width);
      const y = wy(frame.y, world, height);
      const radius = shipRadiusPx(world, width);
      const heading = -frame.theta;

      const vScale = 0.9;
      const vx = wx(frame.x + frame.vx * vScale, world, width);
      const vy = wy(frame.y + frame.vy * vScale, world, height);
      if (Math.hypot(vx - x, vy - y) > 8) {
        drawArrow(ctx, x, y, vx, vy, "#7eb6ff");
      }

      const nose = world.ship_radius * 1.9;
      const tip = world.ship_radius * 3.5;
      ctx.strokeStyle = "#d8deea";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(
        wx(frame.x + Math.cos(frame.theta) * nose, world, width),
        wy(frame.y + Math.sin(frame.theta) * nose, world, height),
      );
      ctx.lineTo(
        wx(frame.x + Math.cos(frame.theta) * tip, world, width),
        wy(frame.y + Math.sin(frame.theta) * tip, world, height),
      );
      ctx.stroke();

      const axial = frame.axial ?? frame.thrust;
      const lateral = frame.lateral ?? 0;
      drawPlume(ctx, x, y, heading, axial, radius);
      drawPlume(ctx, x, y, heading + Math.PI, -axial, radius);
      drawPlume(ctx, x, y, heading - Math.PI / 2, lateral, radius);
      drawPlume(ctx, x, y, heading + Math.PI / 2, -lateral, radius);

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(heading);
      ctx.beginPath();
      ctx.moveTo(radius * 1.6, 0);
      ctx.lineTo(-radius, radius * 0.9);
      ctx.lineTo(-radius * 0.45, 0);
      ctx.lineTo(-radius, -radius * 0.9);
      ctx.closePath();
      ctx.fillStyle = "#f3f1ea";
      ctx.strokeStyle = selection?.kind === "ship" ? "#3ee0c5" : "#c45b3e";
      ctx.lineWidth = selection?.kind === "ship" ? 2.5 : 1.5;
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      if (selection?.kind === "ship") {
        ctx.strokeStyle = "#3ee0c5";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, radius * 2.4, 0, Math.PI * 2);
        ctx.stroke();
      } else if (frame.crash) {
        ctx.strokeStyle = "#e85d4c";
        ctx.beginPath();
        ctx.arc(x, y, radius * 2.2, 0, Math.PI * 2);
        ctx.stroke();
      } else if (frame.success) {
        ctx.strokeStyle = "#3ee0c5";
        ctx.beginPath();
        ctx.arc(x, y, radius * 2.2, 0, Math.PI * 2);
        ctx.stroke();
      }

      drawLegend(ctx);
    };

    render();
    const observer = new ResizeObserver(render);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [world, frames, index, ghost, selection]);

  const pick = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const { world: currentWorld, frames: currentFrames, index: currentIndex } = stateRef.current;
    if (!currentWorld) return null;
    return hitTest(
      currentWorld,
      currentFrames,
      currentIndex,
      canvas.clientWidth,
      canvas.clientHeight,
      event.nativeEvent.offsetX,
      event.nativeEvent.offsetY,
    );
  };

  return (
    <canvas
      ref={canvasRef}
      className="sim-canvas"
      aria-label="Docking scene"
      onMouseMove={(event) => {
        event.currentTarget.style.cursor = pick(event) ? "pointer" : "default";
      }}
      onClick={(event) => onSelect(pick(event))}
    />
  );
}
