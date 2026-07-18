---
name: Clinical-Zen
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#43474f'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#737780'
  outline-variant: '#c3c6d0'
  surface-tint: '#3b6090'
  primary: '#0e3b69'
  on-primary: '#ffffff'
  primary-container: '#2c5282'
  on-primary-container: '#a2c6fd'
  inverse-primary: '#a5c8ff'
  secondary: '#5d5e61'
  on-secondary: '#ffffff'
  secondary-container: '#e2e2e5'
  on-secondary-container: '#636467'
  tertiary: '#3c3a37'
  on-tertiary: '#ffffff'
  tertiary-container: '#53514d'
  on-tertiary-container: '#c8c4bf'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d4e3ff'
  primary-fixed-dim: '#a5c8ff'
  on-primary-fixed: '#001c3a'
  on-primary-fixed-variant: '#204877'
  secondary-fixed: '#e2e2e5'
  secondary-fixed-dim: '#c6c6c9'
  on-secondary-fixed: '#1a1c1e'
  on-secondary-fixed-variant: '#454749'
  tertiary-fixed: '#e6e2dd'
  tertiary-fixed-dim: '#cac6c1'
  on-tertiary-fixed: '#1d1b19'
  on-tertiary-fixed-variant: '#484643'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Source Serif 4
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Source Serif 4
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Source Serif 4
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-sm:
    fontFamily: Source Serif 4
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-metric:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.03em
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-max: 1280px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 48px
  section-gap: 80px
---

## Brand & Style
The design system establishes a "Clinical-Zen" aesthetic, a deliberate fusion of medical-grade precision and high-end wellness tranquility. It targets a sophisticated audience that values both empirical data and mental clarity. 

The style moves away from the cold, sterile visuals of traditional healthcare and the frantic energy of consumer fitness apps. Instead, it adopts a **Corporate Modern** foundation infused with **Tactile** warmth. The interface should feel like a premium physical clinic—ordered, quiet, and exceptionally high-quality. We achieve this through generous whitespace, a grounded color palette, and a "High-End Editorial" approach to information hierarchy.

## Colors
The palette is rooted in stability and organic warmth. **Deep Slate** (#1A1C1E) provides the structural weight and is used for primary text to ensure maximum legibility. **Warm Sand** (#F7F2ED) is the system's "breath"—used for large background areas to soften the clinical feel.

**Trust Blue** (#2C5282) is the functional anchor for primary actions and "Optimal/Low Stress" indicators. The semantic colors—**Amber Gold**, **Terracotta**, and **Soft Mint**—are slightly desaturated to maintain a calm atmosphere, avoiding the aggressive neon tones typical of digital dashboards. Use Soft Mint for recovery metrics and calibration success to reinforce a sense of physical ease.

## Typography
The typographic system utilizes a three-font strategy to balance authority, clarity, and precision. 

1.  **Source Serif 4** is used for all major headings. It provides a literary, authoritative voice that suggests professional expertise. 
2.  **Inter** handles the bulk of the navigational and descriptive content, chosen for its neutral, highly legible character at small sizes.
3.  **JetBrains Mono** is reserved strictly for live biometric data, timestamps, and raw metrics. Its monospaced nature communicates mathematical accuracy and prevents "layout jump" during real-time data updates.

Ensure high contrast ratios (minimum 4.5:1) for all body text against the Warm Sand or white backgrounds.

## Layout & Spacing
The design system employs a **Fluid Grid** with an 8px base unit. To achieve the "Zen" aspect, vertical rhythm is intentional and spacious. 

- **Desktop:** 12-column grid with 48px outer margins. Use a maximum container width of 1280px to prevent line lengths from becoming unreadable. 
- **Mobile:** 4-column grid with 16px outer margins. 
- **Philosophy:** Components should rarely be cramped. Use 80px+ gaps between major sections to allow the eye to rest. For internal component padding, use a minimum of 24px (3 units) to maintain a premium feel.

## Elevation & Depth
Depth is conveyed through **Tonal Layers** and **Ambient Shadows**. Instead of floating elements, the design system uses "Subtle Lift."

- **Level 0 (Base):** Warm Sand (#F7F2ED) surface.
- **Level 1 (Cards):** White (#FFFFFF) surfaces with a very soft, diffused shadow: `0 4px 20px rgba(26, 28, 30, 0.04)`.
- **Level 2 (Interactive/Hover):** White surface with a slightly tighter, darker shadow: `0 8px 30px rgba(26, 28, 30, 0.08)`.

Avoid high-contrast borders or heavy glassmorphism. The goal is to make elements feel like they are gently resting on a soft surface rather than hovering in a digital void.

## Shapes
The shape language is **Rounded**, utilizing a 12px (`0.75rem`) base radius for standard components and 16px (`1rem`) for large containers or cards. This specific range is chosen to feel approachable and "human" without appearing "bubbly" or childish. Buttons and interactive chips should follow the `rounded-lg` (16px) or full pill-shape convention to emphasize touchability.

## Components
- **Buttons:** Primary buttons use Trust Blue with white text, 16px corner radius, and 16px vertical/32px horizontal padding. Secondary buttons use a Deep Slate outline or a Ghost style (no fill).
- **Biometric Cards:** White background, 16px corner radius. The metric (JetBrains Mono) should be prominent, with a small Soft Mint trend indicator for positive health snapshots.
- **Input Fields:** Use a subtle 1px border in a muted Slate (#CBD5E1) on white. Focus states transition the border to Trust Blue with a soft 4px outer glow.
- **Chips/Status Tags:** Use Soft Mint, Amber Gold, or Terracotta with a 10% opacity background of the same color for high-legibility "Low Contrast" indicators. Text inside chips should use the `label-caps` style.
- **Lists:** Clean, border-bottom separation only using a very light neutral gray (#E2E8F0). No zebra-striping; use whitespace to separate items.
- **Progress Bars:** Use a "thick" 8px track height with rounded ends. The track should be a very light version of the status color, and the fill should be the solid status color.