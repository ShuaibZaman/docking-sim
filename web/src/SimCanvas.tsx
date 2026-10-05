import { useEffect, useRef, type MouseEvent } from "react";
import type { Frame, PoseBox, Selection, World } from "./types";

export type GhostTrail = {
  frames: Frame[];
  color: string;
};

type Props = {
  world: World | null;
  frames: Frame[];
  index: number;
  ghosts?: GhostTrail[];
  selection: Selection | null;
  onSelect: (selection: Selection | null) => void;
};

function wx(x: number, world: World, width: number): number {
  return ((x - world.x_min) / (world.x_max - world.x_min)) * width;
}

function wy(y: number, world: World, height: number): number {
  return (1 - (y - world.y_min) / (world.y_max - world.y_min)) * height;
}

function shipRadiusPx(world: World, width: number): number {
  return Math.max(6, (world.ship_radius / (world.x_max - world.x_min)) * width * 1.8);
}

function stationPose(frame: Frame | undefined, world: World): PoseBox {
  return (
    frame?.station ?? {
      cx: world.hull.cx,
      cy: world.hull.cy,
      theta: 0,
      w: world.hull.w,
      h: world.hull.h,
    }
  );
}

function listedPorts(frame: Frame | undefined, world: World): PoseBox[] {
  if (frame?.ports && frame.ports.length) return frame.ports;
  const pose = portPose(frame, world);
  return [{ ...pose, active: true, approach: world.approach_angle + pose.theta }];
}

function activePort(frame: Frame | undefined, world: World): PoseBox {
  const ports = listedPorts(frame, world);
  return ports.find((port) => port.active) ?? ports[0];
}

function portPose(frame: Frame | undefined, world: World): PoseBox {
  return (
    frame?.port_pose ?? {
      cx: world.port.cx,
      cy: world.port.cy,
      theta: 0,
      w: world.port.w,
      h: world.port.h,
    }
  );
}

function worldFromPixel(px: number, py: number, world: World, width: number, height: number) {
  return {
    x: world.x_min + (px / width) * (world.x_max - world.x_min),
    y: world.y_max - (py / height) * (world.y_max - world.y_min),
  };
}

function pointInPose(x: number, y: number, pose: PoseBox): boolean {
  const dx = x - pose.cx;
  const dy = y - pose.cy;
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const lx = c * dx + s * dy;
  const ly = -s * dx + c * dy;
  return Math.abs(lx) <= pose.w / 2 && Math.abs(ly) <= pose.h / 2;
}

function poseCorners(pose: PoseBox): Array<[number, number]> {
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const hw = pose.w / 2;
  const hh = pose.h / 2;
  return [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
  ].map(([lx, ly]) => [pose.cx + c * lx - s * ly, pose.cy + s * lx + c * ly]);
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
  const cursor = worldFromPixel(px, py, world, width, height);
  if (listedPorts(frame, world).some((port) => pointInPose(cursor.x, cursor.y, port))) {
    return { kind: "port" };
  }
  if (pointInPose(cursor.x, cursor.y, stationPose(frame, world))) {
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

function strokePose(
  ctx: CanvasRenderingContext2D,
  pose: PoseBox,
  world: World,
  width: number,
  height: number,
  stroke: string,
  lineWidth: number,
  dashed = false,
) {
  const corners = poseCorners(pose);
  ctx.beginPath();
  corners.forEach(([x, y], i) => {
    const px = wx(x, world, width);
    const py = wy(y, world, height);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawApproachOverlays(
  ctx: CanvasRenderingContext2D,
  world: World,
  frame: Frame | undefined,
  width: number,
  height: number,
) {
  const hull = stationPose(frame, world);
  const dock = activePort(frame, world);
  const approach = dock.approach ?? world.approach_angle + dock.theta;
  const dirX = Math.cos(approach);
  const dirY = Math.sin(approach);
  const length = 6;
  ctx.save();
  ctx.strokeStyle = "rgba(62, 224, 197, 0.55)";
  ctx.lineWidth = 1.25;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(wx(dock.cx - dirX * length, world, width), wy(dock.cy - dirY * length, world, height));
  ctx.lineTo(wx(dock.cx, world, width), wy(dock.cy, world, height));
  ctx.stroke();
  ctx.setLineDash([]);
  strokePose(
    ctx,
    { ...dock, w: dock.w + 0.2, h: dock.h + 0.2 },
    world,
    width,
    height,
    "rgba(62, 224, 197, 0.85)",
    1.25,
    true,
  );
  strokePose(
    ctx,
    {
      ...hull,
      w: hull.w + world.ship_radius * 2,
      h: hull.h + world.ship_radius * 2,
    },
    world,
    width,
    height,
    "rgba(232, 93, 76, 0.45)",
    1,
    true,
  );
  ctx.restore();
}

function drawRelativeInset(
  ctx: CanvasRenderingContext2D,
  world: World,
  frames: Frame[],
  index: number,
  width: number,
  height: number,
) {
  const frame = frames[index];
  if (!frame) return;
  if (width < 240 || height < 180) return;
  const boxW = 104;
  const boxH = 78;
  const left = width - boxW - 10;
  const top = 10;
  ctx.save();
  ctx.fillStyle = "rgba(7, 9, 15, 0.82)";
  ctx.strokeStyle = "#2a3344";
  ctx.lineWidth = 1;
  ctx.fillRect(left, top, boxW, boxH);
  ctx.strokeRect(left, top, boxW, boxH);
  ctx.fillStyle = "#8b93a3";
  ctx.font = "10px 'IBM Plex Sans', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("Target frame", left + 8, top + 6);

  const hull = stationPose(frame, world);
  const dock = activePort(frame, world);
  const toLocal = (x: number, y: number) => {
    const dx = x - hull.cx;
    const dy = y - hull.cy;
    const c = Math.cos(hull.theta);
    const s = Math.sin(hull.theta);
    return [c * dx + s * dy, -s * dx + c * dy] as const;
  };
  const span = 8;
  const plotLeft = left + 8;
  const plotTop = top + 18;
  const plotW = boxW - 16;
  const plotH = boxH - 26;
  const project = (x: number, y: number) => {
    const [lx, ly] = toLocal(x, y);
    return [
      plotLeft + ((lx + span) / (span * 2)) * plotW,
      plotTop + ((span - ly) / (span * 2)) * plotH,
    ] as const;
  };
  const history = frames.slice(0, index + 1);
  if (history.length > 1) {
    ctx.beginPath();
    history.forEach((item, i) => {
      const [px, py] = project(item.x, item.y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "rgba(232, 165, 75, 0.9)";
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }
  const [portX, portY] = project(dock.cx, dock.cy);
  ctx.fillStyle = "#3ee0c5";
  ctx.fillRect(portX - 3, portY - 3, 6, 6);
  const [shipX, shipY] = project(frame.x, frame.y);
  ctx.fillStyle = "#f3f1ea";
  ctx.beginPath();
  ctx.arc(shipX, shipY, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: PoseBox,
  world: World,
  width: number,
  height: number,
  fill: string,
  stroke: string,
  lineWidth: number,
) {
  const corners = poseCorners(pose);
  ctx.beginPath();
  corners.forEach(([x, y], i) => {
    const px = wx(x, world, width);
    const py = wy(y, world, height);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.fill();
  ctx.stroke();
}

function drawShipMark(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  world: World,
  width: number,
  height: number,
  fill: string,
  stroke: string,
) {
  const x = wx(frame.x, world, width);
  const y = wy(frame.y, world, height);
  const radius = shipRadiusPx(world, width) * 0.72;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-frame.theta);
  ctx.beginPath();
  ctx.moveTo(radius * 1.6, 0);
  ctx.lineTo(-radius, radius * 0.9);
  ctx.lineTo(-radius * 0.45, 0);
  ctx.lineTo(-radius, -radius * 0.9);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.4;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawLegend(ctx: CanvasRenderingContext2D, showRocks: boolean) {
  const items: Array<[string, string, "box" | "line"]> = [
    ["#f3f1ea", "Ship", "box"],
    ["#3ee0c5", "Port", "box"],
    ["#8b93a3", "Hull", "box"],
    ["#e8a54b", "Path", "line"],
    ["#7eb6ff", "Velocity", "line"],
  ];
  if (showRocks) items.splice(3, 0, ["#968872", "Rock", "box"]);
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

export function SimCanvas({ world, frames, index, ghosts, selection, onSelect }: Props) {
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

      const shown = frames[index];
      const hullPose = stationPose(shown, world);
      const ports = listedPorts(shown, world);
      const dockPose = activePort(shown, world);
      (shown?.asteroids ?? []).forEach((rock) => {
        ctx.beginPath();
        ctx.arc(
          wx(rock.x, world, width),
          wy(rock.y, world, height),
          Math.max(4, (rock.r / (world.x_max - world.x_min)) * width),
          0,
          Math.PI * 2,
        );
        ctx.fillStyle = "#968872";
        ctx.fill();
      });

      drawApproachOverlays(ctx, world, shown, width, height);
      const hullSelected = selection?.kind === "station";
      drawPose(ctx, hullPose, world, width, height, "#2a3344", hullSelected ? "#f3f1ea" : "#8b93a3", hullSelected ? 2.5 : 1.5);
      ctx.fillStyle = "#d8deea";
      ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("STATION", wx(hullPose.cx, world, width), wy(hullPose.cy, world, height));
      if (shown?.hit_hull) {
        ctx.fillStyle = "#e85d4c";
        ctx.font = "10px 'IBM Plex Sans', sans-serif";
        ctx.fillText("collision", wx(hullPose.cx, world, width), wy(hullPose.cy, world, height) + 14);
      }

      const portSelected = selection?.kind === "port";
      ports.forEach((port) => {
        const live = Boolean(port.active);
        drawPose(
          ctx,
          port,
          world,
          width,
          height,
          live ? "rgba(62, 224, 197, 0.22)" : "rgba(62, 224, 197, 0.06)",
          live ? (portSelected ? "#f3f1ea" : "#3ee0c5") : "rgba(62, 224, 197, 0.45)",
          live ? (portSelected ? 2.5 : 1.5) : 1,
        );
      });
      const speedLimit = world.dock_speed_max ?? 0.08;
      const angleLimit = world.dock_angle_max_deg ?? 12;
      const portLabelX = wx(dockPose.cx, world, width);
      const portLabelY = wy(dockPose.cy, world, height);
      ctx.fillStyle = "#3ee0c5";
      ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
      ctx.fillText(ports.length > 1 ? "TARGET" : "PORT", portLabelX, portLabelY + 22);
      ctx.font = "10px 'IBM Plex Sans', sans-serif";
      ctx.fillStyle = "#8b93a3";
      ctx.fillText(
        `speed \u2264 ${speedLimit}  \u00b7  \u00b1${Math.round(angleLimit)}\u00b0`,
        portLabelX,
        portLabelY + 36,
      );

      (ghosts ?? []).forEach((ghost) => {
        if (ghost.frames.length < 2) return;
        const ghostEnd = Math.min(index, ghost.frames.length - 1);
        const ghostTrail = ghost.frames.slice(0, ghostEnd + 1);
        ctx.beginPath();
        ghostTrail.forEach((frame, i) => {
          const x = wx(frame.x, world, width);
          const y = wy(frame.y, world, height);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = ghost.color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        const ghostFrame = ghost.frames[ghostEnd];
        if (ghostFrame) drawShipMark(ctx, ghostFrame, world, width, height, ghost.color, ghost.color);
      });

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
        drawLegend(ctx, false);
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

      drawRelativeInset(ctx, world, frames, index, width, height);
      drawLegend(ctx, (frame.asteroids?.length ?? 0) > 0);
    };

    render();
    const observer = new ResizeObserver(render);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [world, frames, index, ghosts, selection]);

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
