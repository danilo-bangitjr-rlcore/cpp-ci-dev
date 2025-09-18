export interface TimeState {
  hours: number;
  minutes: number;
  seconds: number;
}

interface TimeInputProps {
  value: TimeState;
  onChange: (value: TimeState) => void;
}

export default function TimeInput({ value, onChange }: TimeInputProps) {
  const handleChange = (field: keyof TimeState, newValue: string) => {
    const num = parseInt(newValue, 10);
    if (isNaN(num)) return;

    let clamped = num;
    if (field === 'hours') {
      clamped = Math.max(0, Math.min(168, num));
    } else {
      clamped = Math.max(0, Math.min(59, num));
    }

    onChange({ ...value, [field]: clamped });
  };

  return (
    <div className="flex items-center space-x-1">
      <input
        type="number"
        value={value.hours}
        onChange={(e) => handleChange('hours', e.target.value)}
        min="0"
        max="168"
        className="w-16 border border-gray-300 rounded text-center"
        placeholder="HH"
      />
      <span>:</span>
      <input
        type="number"
        value={value.minutes}
        onChange={(e) => handleChange('minutes', e.target.value)}
        min="0"
        max="59"
        className="w-16 border border-gray-300 rounded text-center"
        placeholder="MM"
      />
      <span>:</span>
      <input
        type="number"
        value={value.seconds}
        onChange={(e) => handleChange('seconds', e.target.value)}
        min="0"
        max="59"
        className="w-16 border border-gray-300 rounded text-center"
        placeholder="SS"
      />
    </div>
  );
}
