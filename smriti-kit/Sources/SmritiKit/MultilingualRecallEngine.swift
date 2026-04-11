import Foundation
import NaturalLanguage

@MainActor
public class MultilingualRecallEngine: ObservableObject {

    public struct NarratedRecallResult {
        public let items: [SmritiRecallItem]
        public let narration: String?
        public let detectedLanguageCode: String?
        
        public init(items: [SmritiRecallItem], narration: String?, detectedLanguageCode: String?) {
            self.items = items
            self.narration = narration
            self.detectedLanguageCode = detectedLanguageCode
        }
    }

    private let gemma = GemmaModelManager.shared

    public init() {}

    public func query(_ text: String) async throws -> NarratedRecallResult {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        let detected = recognizer.dominantLanguage

        var searchText = text
        if let lang = detected, lang != .english {
            let translationPrompt = """
            Translate this memory search query to English for semantic search. Preserve all specificity. Query: \(text)
            Return only the English translation. No explanation.
            """
            
            if let result = try? await gemma.generate(prompt: translationPrompt, maxTokens: 64) {
                searchText = result.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        let api = SmritiAPI()
        let request = SmritiRecallRequest(query: searchText, session_id: "smriti-query", top_k: 20, person_filter: nil, location_filter: nil, time_start: nil, time_end: nil, min_confidence: 0)
        let response = try await api.recall(request)
        let items = response.results

        var narration: String? = nil
        if !items.isEmpty && gemma.isAvailable() {
            let topItems = items.prefix(3)
            struct SummaryItem: Encodable { let label: String; let surprise: Double }
            let summaryItems = topItems.map { item in
                SummaryItem(label: item.primary_description, surprise: item.effectiveSurpriseScore)
            }
            if let jsonData = try? JSONEncoder().encode(summaryItems),
               let jsonStr = String(data: jsonData, encoding: .utf8) {
                
                let displayLanguage = Locale.current.localizedString(forLanguageCode: detected?.rawValue ?? "en") ?? "English"
                
                let narrationPrompt = """
                You are Smriti, a private memory assistant.
                Narrate these memory search results in \(displayLanguage) in 2-3 sentences.
                Focus on what was surprising, what persisted, and what changed.
                Do not mention scores, identifiers, or technical terms.
                Results: \(jsonStr)
                """
                
                narration = try? await gemma.generate(prompt: narrationPrompt, maxTokens: 128)
            }
        }
        
        return NarratedRecallResult(items: items, narration: narration?.trimmingCharacters(in: .whitespacesAndNewlines), detectedLanguageCode: detected?.rawValue)
    }
}
