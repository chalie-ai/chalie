# Chalie Web Interface Specification

The Chalie web interface is a collection of three single-page applications: the main chat interface, the cognitive dashboard ("Brain"), and the onboarding wizard. All follow the **Radiant design system** for a cinematic, restrained dark UI.

## Design System: Radiant

The visual language is inspired by "blockbuster dark UI" (JARVIS, Tron Legacy, K-pop demon hunter HUDs). The canvas is near-black. Color exists only as atmospheric light (distant orbs on canvas) and precision accents (thin luminous edges, single-color glows on interactive elements).

### Core Design Principles

**Darkness as Canvas**
- Base color: `#06080e` (near-black, darker than most dark modes)
- Surfaces: `rgba(255, 255, 255, 0.03)` with `rgba(255, 255, 255, 0.07)` borders
- No purple-tinted surfaces — color bleeds from canvas atmosphere only
- Primary accent: `#8A5CFF` (neon violet) for buttons, active borders, focus states
- Secondary accent: `#FF2FD1` (plasma magenta) for presence indicators
- Tertiary accent: `#00F0FF` (electric cyan) for processing states

**Precision Over Diffusion**
- One glow color per element, one thin edge highlight
- No rainbow gradients on small elements
- No stacked multi-color box-shadows
- No `inset` box-shadows for decoration
- Avoid high-alpha accent fills (keep below 0.08)

**Restraint as Luxury**
- When nothing glows, the newest Chalie message's thin violet edge catches the eye
- If everything glowed, nothing would
- Buttons use solid `#8A5CFF` — glow only appears on hover
- Transitions use `220ms ease`

**Atmospheric Depth**
- Canvas renders 4 orbs at very low alpha (0.05–0.08) drifting over 25–35s cycles
- Two warm (violet, magenta), two cool (cyan, indigo) for natural color temperature
- Provides color context without competing with UI elements

### Color Palette

**Backgrounds & Surfaces**
- Floor: `#06080e`
- Surfaces: `rgba(255, 255, 255, 0.025–0.05)`
- Borders: `rgba(255, 255, 255, 0.06–0.07)`
- Grain overlay: `opacity: 0.04; mix-blend-mode: overlay`

**Accents**
- Violet (primary): `#8A5CFF`
- Magenta (secondary): `#FF2FD1`
- Cyan (tertiary): `#00F0FF`

**Text**
- Primary: `#eae6f2`
- Secondary: `rgba(234, 230, 242, 0.58)`
- Tertiary: `rgba(234, 230, 242, 0.30)`

### Implementation Guardrails

- No purple-tinted surfaces — color bleeds from canvas atmosphere only
- One glow color per element, no stacked multi-color box-shadows
- No fast ambient motion (25–35s drift minimum)
- `line-height: 1.6`, all transitions `220ms ease`
- See `frontend/interface/style.css` for exact values

## Layout Structure

### Title Bar (60px, fixed)
- Centered app name or section title
- Optional status indicators
- Optional media controls (when voice playing)
- Optional navigation (Brain icon, settings)

### Chat Area (scrollable middle)
- Messages in chronological order
- System messages (Chalie) on left with thin violet top-edge gradient
- User messages on right with distinct background
- Support for cards: scheduled items, lists, etc.
- Scroll depth fade at top/bottom

### Prompt Box (80px, fixed bottom)
- Text input field (full width, `line-height: 1.6`)
- Left side: Microphone button (voice input)
- Right side: Send button
- Visual feedback while processing

## Presence Dot States

All presence dots include soft halo: `box-shadow: 0 0 8px currentColor`

- **Resting** (breathing): Magenta `#FF2FD1`, scale animation
- **Processing** (pulse): Cyan `#00F0FF`, scale animation
- **Thinking** (glow): Violet `#8A5CFF`, variable-intensity glow
- **Retrieving Memory** (ripple): Cyan with expanding ripple shadow
- **Planning** (shimmer): Gradient cycling violet → cyan
- **Responding** (waveform): Amber bar with waveform animation

## Active Message Treatment

When Chalie's newest message is active:
- Border transitions to `rgba(138, 92, 255, 0.30)`
- Thin top-edge gradient: `transparent → violet → cyan → transparent`
- Subtle outer glow: `0 0 20px rgba(138, 92, 255, 0.07)`
- 400ms transition on border-color and box-shadow

## Canvas Atmosphere

A background `<canvas>` renders 4 low-alpha orbs (two warm, two cool) drifting on slow 25–35s cycles. Provides cinematic depth without competing with UI elements. See `frontend/interface/ambient_canvas.js` for implementation.

## Cards System

Reusable card components render structured data:

**Scheduled Item Cards**
- Title, description, scheduled time
- Recurrence indicator
- Status indicator (pending, completed)
- Edit/delete actions

**List Cards**
- List name and type
- Item count
- Recent items preview
- Add/manage actions

**Goal Cards**
- Goal title and progress bar
- Target date
- Status (active, completed, abandoned)
- Update actions

**Knowledge Cards**
- Concept name and strength
- Related concepts
- Last accessed time

All cards use the same design language: dark surfaces, violet accents, thin borders.

## Voice I/O (Optional)

### Speech-to-Text (Microphone Button)
- Click to start/stop recording
- Visual feedback during recording (waveform animation)
- Transcribed text appears in prompt box
- User clicks send to submit (not automatic)

### Text-to-Speech (Speaker Icon)
- Speaker icon appears below Chalie messages
- Click opens audio player in title bar
- Player shows: play/pause, close
- Audio plays immediately
- Allows listening while multitasking

## Applications

### 1. `frontend/interface/` — Main Chat UI
- Layout: title bar (60px) + chat (scrollable) + prompt (80px)
- Presence dot and status indicators
- Canvas atmosphere rendering
- Card support for lists, goals, schedules
- Voice I/O optional
- Home view shows recent conversations

### 2. `frontend/brain/` — Cognitive Dashboard
- Admin view of memory system
- Episodic memories with decay visualization
- Semantic concepts and relationships
- Routing decision audit trail
- Tool execution history
- Settings and configuration
- Tool management interface

### 3. `frontend/on-boarding/` — Account Setup Wizard
- Multi-step account creation
- LLM provider configuration
- Voice endpoint setup (optional)
- Tool configuration
- Welcome screen after setup

## Responsive Design

- **Mobile**: Full-width, touch-friendly buttons, portrait optimized
- **Tablet**: Increased spacing, wider chat area
- **Desktop**: Centered with max-width, comfortable spacing

All three sizes maintain Radiant design fidelity.

## References

See `CLAUDE.md` "Design Philosophy: Radiant" for the authoritative design specification. This document is a UI-specific interpretation of that design system.
