import { useState } from 'react';
import TimeInput from './TimeInput';

interface TimeState {
  hours: number;
  minutes: number;
  seconds: number;
}

export default function GeneralSettings() {
  const [observationPeriod, setObservationPeriod] = useState<TimeState>({ hours: 0, minutes: 0, seconds: 0 });
  const [actionPeriod, setActionPeriod] = useState<TimeState>({ hours: 0, minutes: 0, seconds: 0 });
  const [effectHorizon, setEffectHorizon] = useState<TimeState>({ hours: 0, minutes: 0, seconds: 0 });

  const convertToIsoDuration = (time: TimeState): string => {
    const hours = time.hours > 0 ? `${time.hours}H` : '';
    const minutes = time.minutes > 0 ? `${time.minutes}M` : '';
    const seconds = time.seconds > 0 ? `${time.seconds}S` : '';
    return `PT${hours}${minutes}${seconds}` || 'PT0S';
  };

  const handleSubmit = () => {
    const data = {
      observationPeriod: convertToIsoDuration(observationPeriod),
      actionPeriod: convertToIsoDuration(actionPeriod),
      effectHorizon: convertToIsoDuration(effectHorizon),
    };
    console.log('Submitting general settings:', data);
  };

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-lg font-semibold">General Settings</h2>

      <div className="space-y-2">
        <label className="block text-sm font-medium">Observation Period</label>
        <TimeInput value={observationPeriod} onChange={setObservationPeriod} />
        <p className="text-xs text-gray-500">Format: hh:mm:ss</p>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium">Action Period</label>
        <TimeInput value={actionPeriod} onChange={setActionPeriod} />
        <p className="text-xs text-gray-500">Format: hh:mm:ss</p>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium">Effect Horizon</label>
        <TimeInput value={effectHorizon} onChange={setEffectHorizon} />
        <p className="text-xs text-gray-500">Format: hh:mm:ss</p>
      </div>

      <button
        onClick={handleSubmit}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Submit
      </button>
    </div>
  );
}
