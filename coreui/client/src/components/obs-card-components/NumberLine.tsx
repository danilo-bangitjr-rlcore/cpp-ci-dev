import React from 'react';

type Range = {
  label: string;
  values: [number, number];
  color: string;
};

type NumberLineProps = {
  ranges: Range[];
  min?: number;
  max?: number;
};

export const NumberLine: React.FC<NumberLineProps> = ({
  ranges,
  min = 0,
  max = 100,
}) => {
  const toPercent = (value: number) => ((value - min) / (max - min)) * 100;

  return (
    <div style={{ marginTop: 20 }}>
      <div
        style={{
          position: 'relative',
          height: '50px',
          background: '#ddd',
          borderRadius: '3px',
        }}
      >
        {ranges.map(({ values, color, label }) =>
          values.map((val, idx) => {
            const percent = toPercent(val);
            return (
              <div
                key={`${label}-${idx}`}
                title={`${label} ${idx === 0 ? 'low' : 'high'}: ${val}`}
                style={{
                  position: 'absolute',
                  left: `${percent}%`,
                  top: '-6px',
                  width: '5px',
                  height: '65px',
                  background: color,
                  border: color === 'red' ? '1px dashed red' : undefined,
                }}
              />
            );
          })
        )}
      </div>

      {/* Axis labels */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '15px',
          marginTop: '4px',
        }}
      >
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};
