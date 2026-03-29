import type { SmritiPersonJournal as JournalType } from "../../types";

type PersonJournalProps = {
  personName: string;
  onPersonNameChange: (value: string) => void;
  journal: JournalType | null;
  onLoadJournal: (name: string) => Promise<void>;
};

export default function PersonJournal({
  personName,
  onPersonNameChange,
  journal,
  onLoadJournal,
}: PersonJournalProps) {
  return (
    <div className="smriti-stack">
      <section className="panel panel--stable">
        <div className="smriti-panel-header">
          <div>
            <p className="eyebrow">Person Journal</p>
            <h3>Entity timeline</h3>
          </div>
          <p className="muted">Inspect every confirmed media link for a named person, newest first.</p>
        </div>
        <form
          className="smriti-search-form"
          onSubmit={async (event) => {
            event.preventDefault();
            await onLoadJournal(personName);
          }}
        >
          <input
            value={personName}
            onChange={(event) => onPersonNameChange(event.target.value)}
            placeholder="Enter a person name"
          />
          <button type="submit" className="primary" disabled={!personName.trim()}>
            Load journal
          </button>
        </form>
      </section>

      <section className="smriti-journal-grid">
        {journal?.entries.length ? journal.entries.map((entry) => (
          <article key={entry.media_id} className="panel smriti-journal-entry">
            <p className="eyebrow">{new Date(entry.ingested_at).toLocaleString()}</p>
            <h4>{entry.media_id}</h4>
            <p className="muted">{entry.file_path}</p>
          </article>
        )) : (
          <div className="panel smriti-empty-state">
            <p className="eyebrow">No Journal Loaded</p>
            <h4>Search for a tagged person</h4>
            <p className="muted">After you confirm a person in Deepdive, Smriti propagates the label to similar media and exposes the resulting journal here.</p>
          </div>
        )}
      </section>
    </div>
  );
}
