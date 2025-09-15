import React from 'react';

type RangeInputProps = {
  label: string;
  color: string;
  value: [number, number];
  onChange: (value: [number, number]) => void;
  valid: boolean;
};

export const RangeInput: React.FC<RangeInputProps> = ({
  label,
  color,
  value,
  onChange,
  valid,
}) => {
  const handleChange = (index: 0 | 1, v: number) => {
    const next: [number, number] = [...value] as [number, number];
    next[index] = v;
    onChange(next);
  };

  return (
    <div>
      <h4 style={{ color }}>{label}</h4>
      <div style={{ display: 'flex', gap: '8px' }}>
        <input
          type="number"
          value={value[0] ?? ''}
          style={{
            border: '1px solid black',
            maxWidth: '60px',
            textAlign: 'center',
          }}
          onChange={(e) => handleChange(0, Number(e.target.value) || 0)}
        />
        <input
          type="number"
          value={value[1] ?? ''}
          style={{
            border: '1px solid black',
            maxWidth: '60px',
            textAlign: 'center',
          }}
          onChange={(e) => handleChange(1, Number(e.target.value) || 0)}
        />
      </div>
      {!valid && (
        <span style={{ color: 'red', fontSize: '12px' }}>
          Low must be ≤ High
        </span>
      )}
    </div>
  );
};
