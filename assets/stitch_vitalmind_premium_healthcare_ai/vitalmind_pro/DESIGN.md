---
name: VitalMind Pro
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
  on-surface-variant: '#424656'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#727687'
  outline-variant: '#c2c6d8'
  surface-tint: '#0054d6'
  primary: '#0050cb'
  on-primary: '#ffffff'
  primary-container: '#0066ff'
  on-primary-container: '#f8f7ff'
  inverse-primary: '#b3c5ff'
  secondary: '#006875'
  on-secondary: '#ffffff'
  secondary-container: '#00e3fd'
  on-secondary-container: '#00616d'
  tertiary: '#6834d2'
  on-tertiary: '#ffffff'
  tertiary-container: '#8252ec'
  on-tertiary-container: '#fdf6ff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae1ff'
  primary-fixed-dim: '#b3c5ff'
  on-primary-fixed: '#001849'
  on-primary-fixed-variant: '#003fa4'
  secondary-fixed: '#9cf0ff'
  secondary-fixed-dim: '#00daf3'
  on-secondary-fixed: '#001f24'
  on-secondary-fixed-variant: '#004f58'
  tertiary-fixed: '#e9ddff'
  tertiary-fixed-dim: '#d0bcff'
  on-tertiary-fixed: '#23005c'
  on-tertiary-fixed-variant: '#5516be'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-hero:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.04em
  display-hero-mobile:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
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
  label-caps:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.08em
  metric-xl:
    fontFamily: Geist
    fontSize: 64px
    fontWeight: '700'
    lineHeight: 64px
    letterSpacing: -0.05em
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  unit: 8px
  container-max-width: 1440px
  gutter: 24px
  margin-desktop: 64px
  margin-mobile: 20px
  stack-sm: 12px
  stack-md: 24px
  stack-lg: 48px
---

## Brand & Style
The design system is engineered to bridge the gap between high-stakes clinical precision and restorative mental wellness. It targets executive and high-performance users who require a premium, medical-grade SaaS experience that feels like a concierge health service rather than a sterile utility.

The visual direction is **Minimal Luxury AI**. It employs a sophisticated blend of **Glassmorphism** and **Modern Corporate** aesthetics, utilizing translucent layers and frosted-glass surfaces to evoke a sense of clarity and breathability. The atmosphere is one of calm authority—professional, trustworthy, and deeply intuitive. High-end mesh gradients provide a sense of organic movement, mirroring the fluid nature of biometrics and mental states.

## Colors
This design system utilizes a high-fidelity palette designed for data visualization and emotional regulation.

- **Primary (Medical Blue):** Used for core actions, primary brand elements, and confirmed medical states.
- **Secondary (Soft Cyan):** Applied to AI-driven insights and active monitoring states.
- **Accent (AI Purple):** Reserved for predictive analytics and machine learning features.
- **Surface Strategy:** In Light Mode, use subtle mesh gradients of Cyan and Purple at 5% opacity on a #F8FAFC base. In Dark Mode, use Deep Navy (#0F172A) with elevated glass surfaces.
- **Semantic Colors:** Emerald, Amber, and Coral Red follow standard medical status protocols but are softened with higher luminance to maintain the "calm" aesthetic.

## Typography
The typography system prioritizes data legibility and rhythmic hierarchy. **Geist** is used for headlines, labels, and large numeric metrics to provide a technical, monospaced-adjacent precision. **Inter** is utilized for body copy to ensure maximum readability during long-form health reports.

**Metric-XL** is specifically designed for stress scores and heart rate variability (HRV) data, creating a focal point in dashboard views. Ensure high-contrast ratios for all body text, while using `label-caps` in neutral-500 for secondary metadata.

## Layout & Spacing
The layout follows a **fluid grid** model with generous margins to mimic the "white space" of luxury editorial design.

- **Grid:** A 12-column layout on desktop, transitioning to a 1-column stack on mobile.
- **Rhythm:** An 8px linear scale is used for all internal padding and alignment.
- **Sectioning:** Content is grouped into logical "Focus Zones." Use `stack-lg` (48px) to separate primary modules like the Stress Graph and the Recommendation Engine.
- **Safe Areas:** Maintain a minimum of 24px gutter between glassmorphic cards to ensure the background blurs do not visually bleed into one another.

## Elevation & Depth
Depth is created through transparency and refraction rather than traditional dark shadows.

- **Glass Surfaces:** Cards use a `backdrop-filter: blur(20px)` with a semi-transparent white (80% opacity) or navy (70% opacity) fill.
- **Stroke:** Apply a 1px solid border at 10% opacity (Primary Blue or White) to define card edges without adding visual weight.
- **Shadows:** Use a single "Ambient Glow"—a very large, soft shadow (Blur: 40px, Spread: -10px) with a 5% opacity tint of the Primary Blue.
- **Z-Index Strategy:** Floating action panels sit at the highest elevation, using a more intense blur (40px) to signify priority.

## Shapes
The shape language is defined by oversized, hyper-rounded corners that evoke a sense of safety and modern hardware design (reminiscent of premium wearables).

- **Base Radius:** Containers and primary cards use a `24px` (rounded-xl) radius.
- **Interactive Elements:** Buttons and input fields use a pill-shaped (`999px`) or `16px` radius to maintain a soft touch.
- **Inner Padding:** Ensure internal padding is at least 50% of the corner radius to prevent visual "pinching" of content.

## Components
- **Glass Cards:** The primary container. Features a subtle 1px border and 24px rounded corners. Header areas within cards should be separated by a hairline stroke (10% opacity).
- **Primary Buttons:** High-contrast Medical Blue fill with white text. Apply a subtle internal glow (top-down linear gradient) to give a tactile, "pressed" look.
- **Metric Widgets:** Features a large `metric-xl` value, a Geist-label, and a small sparkline showing a 24-hour trend.
- **Status Chips:** Small, pill-shaped indicators with a 10% opacity background of the status color (e.g., Emerald) and 100% opacity text.
- **Input Fields:** Semi-transparent backgrounds with a 2px "active" border that glows when focused using the AI Purple.
- **Progress Bars:** Dual-layered; a semi-transparent track with a glowing, gradient-filled foreground (Primary to Secondary). Use `transition: width 1s ease-out` for AI-calculated updates.
- **Floating Action Panel:** A bottom-docked navigation or tool bar with a heavy backdrop blur and high-contrast icons.