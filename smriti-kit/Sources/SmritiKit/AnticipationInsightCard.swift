import SwiftUI

public struct AnticipationInsightCard: View {
    let insight: String
    let onDismiss: () -> Void
    let onSeePattern: () -> Void
    
    public init(insight: String, onDismiss: @escaping () -> Void, onSeePattern: @escaping () -> Void) {
        self.insight = insight
        self.onDismiss = onDismiss
        self.onSeePattern = onSeePattern
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Pattern Noticed", systemImage: "sparkles")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(Color(red: 0.235, green: 0.765, blue: 0.765)) // smritiTeal
                Spacer()
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                #if os(macOS)
                .buttonStyle(.plain)
                #else
                .buttonStyle(.borderless)
                #endif
            }
            
            Text(insight)
                .font(.subheadline)
                .foregroundColor(.primary)
                .fixedSize(horizontal: false, vertical: true)
                .lineSpacing(4)
            
            Button("View journal") {
                onSeePattern()
            }
            .font(.caption.weight(.semibold))
            .padding(.vertical, 6)
            .padding(.horizontal, 12)
            .background(Color(red: 0.4196, green: 0.3607, blue: 0.9058).opacity(0.2)) // smritiAccent
            .foregroundColor(.white)
            .clipShape(Capsule())
            #if os(macOS)
            .buttonStyle(.plain)
            #else
            .buttonStyle(.borderless)
            #endif
        }
        .padding(16)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.white.opacity(0.08), lineWidth: 0.5)
        )
    }
}
