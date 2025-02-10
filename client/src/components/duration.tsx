import React, { useState } from "react";
import { Temporal } from "temporal-polyfill";
import { Field, FieldGroup, Label } from "./fieldset";
import { Input } from "./input";

interface DurationInputProps {
  onChange: (isoDuration: string) => void;
  defaultValue?: string;
  units?: (keyof Temporal.DurationLike)[];
}

const DurationInput: React.FC<DurationInputProps> = ({
  onChange,
  defaultValue,
  units = ["hours", "minutes", "seconds"],
}) => {
  const rawDuration = defaultValue
    ? Temporal.Duration.from(defaultValue)
    : new Temporal.Duration();
  const [durationParts, setDurationParts] = useState<Temporal.DurationLike>({
    years: rawDuration.years,
    months: rawDuration.months,
    weeks: rawDuration.weeks,
    days: rawDuration.days,
    hours: rawDuration.hours,
    minutes: rawDuration.minutes,
    seconds: rawDuration.seconds,
    microseconds: rawDuration.microseconds,
    milliseconds: rawDuration.milliseconds,
  });

  const handleChange = (value: string, unit: keyof Temporal.DurationLike) => {
    const parsedValue = parseInt(value, 10);
    if (!isNaN(parsedValue) && parsedValue >= 0) {
      updateDuration(unit, parsedValue);
    }
  };

  const updateDuration = (unit: keyof Temporal.DurationLike, value: number) => {
    setDurationParts({ ...durationParts, [unit]: value });
    const rawDuration = Temporal.Duration.from({
      ...durationParts,
      [unit]: value,
    });
    onChange(rawDuration.toString());
  };

  const capitalizeFirstLetter = (val: string) =>
    String(val).charAt(0).toUpperCase() + String(val).slice(1);

  return (
    <FieldGroup className="grid grid-cols-3 grid-gap-2">
      {units.map((unit) => (
        <Field className="flex flex-col" key={unit}>
          <Label>{capitalizeFirstLetter(unit)}</Label>
          <Input
            name={unit}
            type="number"
            defaultValue={durationParts[unit]}
            onChange={(e) => handleChange(e.target.value, unit)}
            min="0"
          />
        </Field>
      ))}
    </FieldGroup>
  );
};

export default DurationInput;
