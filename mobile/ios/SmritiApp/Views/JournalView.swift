import SwiftUI
import SmritiKit
import UIKit

struct JournalView: View {
    @StateObject private var appModel = SmritiAppModel.shared
    @State private var selectedDate = Date()
    @State private var animatedTextOpacity: Double = 0
    @State private var showAnimation = false
    @State private var textWords: [String] = []
    
    private let engine = SilentJournalEngine()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Top circle nav
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 16) {
                        ForEach(-6...0, id: \.self) { dayOffset in
                            let date = Calendar.current.date(byAdding: .day, value: dayOffset, to: Date()) ?? Date()
                            let isSelected = Calendar.current.isDate(date, inSameDayAs: selectedDate)
                            let hasEntry = engine.cachedJournal(for: date) != nil
                            
                            VStack(spacing: 4) {
                                Text("\(Calendar.current.component(.day, from: date))")
                                    .font(.system(size: 15))
                                    .frame(width: 32, height: 32)
                                    .background(isSelected ? Color.smritiAccent : Color.clear)
                                    .foregroundColor(isSelected ? .white : .primary)
                                    .clipShape(Circle())
                                    .onTapGesture {
                                        withAnimation(.smritiSpring) {
                                            selectedDate = date
                                        }
                                    }
                                
                                Circle()
                                    .fill(hasEntry ? Color.smritiAccent : Color.clear)
                                    .frame(width: 4, height: 4)
                            }
                        }
                    }
                    .padding()
                }
                .background(Color.smritiSurface)
                
                Divider().background(Color.smritiDivider)
                
                // Body
                ZStack {
                    if let entry = engine.cachedJournal(for: selectedDate) {
                        ScrollView {
                            Text(entry)
                                .font(.system(size: 17, weight: .regular))
                                .lineSpacing(8)
                                .padding(24)
                                .opacity(showAnimation ? 1 : 0)
                                .animation(.easeIn(duration: 0.5), value: showAnimation)
                                .onAppear {
                                    let fmt = DateFormatter()
                                    fmt.dateFormat = "yyyyMMdd"
                                    let key = "smriti.journal.viewed.\(fmt.string(from: selectedDate))"
                                    if !UserDefaults.standard.bool(forKey: key) {
                                        showAnimation = false
                                        withAnimation(.easeIn(duration: 0.8)) {
                                            showAnimation = true
                                        }
                                        UserDefaults.standard.set(true, forKey: key)
                                    } else {
                                        showAnimation = true
                                    }
                                }
                        }
                    } else if Calendar.current.isDateInToday(selectedDate) && Calendar.current.component(.hour, from: Date()) < 21 {
                        Text("Entry will appear tonight.")
                            .foregroundColor(.secondary)
                            .font(.callout)
                    } else if Calendar.current.isDateInToday(selectedDate) {
                        VStack(spacing: 16) {
                            TimelineView(.animation) { timeline in
                                let t = timeline.date.timeIntervalSinceReferenceDate
                                HStack(spacing: 6) {
                                    ForEach(0..<5, id: \.self) { i in
                                        let h = 12 + 8 * sin(t * 2 + Double(i))
                                        Rectangle()
                                            .fill(Color.smritiAccent.opacity(0.3))
                                            .frame(width: 4, height: h)
                                            .cornerRadius(2)
                                    }
                                }
                            }
                            Text("Writing tonight's entry…")
                                .foregroundColor(.secondary)
                                .font(.callout)
                        }
                    } else {
                        Text("No entry for this day.")
                            .foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .navigationTitle("Journal")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        shareJournal()
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .disabled(engine.cachedJournal(for: selectedDate) == nil)
                }
            }
        }
        .sheet(item: $shareURLItem) { urlItem in
            ActivityView(activityItems: [urlItem.url])
        }
        .overlay {
            if isGenerating {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                ProgressView()
                    .tint(.white)
            }
        }
    }
    
    // Feature 5 wire point
    @State private var isGenerating = false
    @State private var shareURLItem: URLItem?
    
    private struct URLItem: Identifiable {
        let id = UUID()
        let url: URL
    }
    
    private func shareJournal() {
        guard let entry = engine.cachedJournal(for: selectedDate) else { return }
        isGenerating = true
        Task {
            let generator = MemoryCardGenerator()
            if let cardURL = generator.generateMemoryCard(imagePath: nil, date: selectedDate, summary: entry) {
                DispatchQueue.main.async {
                    isGenerating = false
                    shareURLItem = URLItem(url: cardURL)
                }
            } else {
                DispatchQueue.main.async {
                    isGenerating = false
                }
            }
        }
    }
    
    private func saveToTempFile(_ data: Data, name: String) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        try? data.write(to: url)
        return url
    }
}

private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: applicationActivities)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
