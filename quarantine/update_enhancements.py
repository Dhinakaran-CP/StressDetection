import re

with open('frontend/src/components/ResultEnhancements.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''  const analysis = useMemo(() => {
    const individual = result?.individual_predictions || {};

    const points = [];
    if (individual.facial !== null && individual.facial !== undefined) {
      points.push({ key: "facial", label: "Facial", value: toPercent(individual.facial), reason: "facial tension" });
    }
    if (individual.voice !== null && individual.voice !== undefined) {
      points.push({ key: "voice", label: "Voice", value: toPercent(individual.voice), reason: "vocal strain" });
    }
    if (individual.physiological !== null && individual.physiological !== undefined) {
      points.push({ key: "physiological", label: "Physiological", value: toPercent(individual.physiological), reason: "elevated physiological signals" });
    }

    // Fallback if no individual predictions exist (should not happen if backend works)
    if (points.length === 0) {
      points.push({ key: "overall", label: "Overall", value: stressPercentage, reason: "overall stress factors" });
    }

    const sorted = [...points].sort((a, b) => b.value - a.value);
    const total = points.reduce((sum, item) => sum + item.value, 0) || 1;

    return {
      cause: sorted.length > 1 ? \\ and \\ : sorted[0].reason,
      contributions: points.map((item) => ({
        ...item,
        contribution: Math.round((item.value / total) * 100),
        band: toStressBand(item.value),
      })),
    };
  }, [result, stressPercentage]);'''

# We need to replace the analysis useMemo
content = re.sub(r'  const analysis = useMemo\(\(\) => \{.*?    \};\n  \}, \[result, stressPercentage\]\);', replacement, content, flags=re.DOTALL)

with open('frontend/src/components/ResultEnhancements.jsx', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated ResultEnhancements.jsx")
