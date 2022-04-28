import UIKit

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()

    private lazy var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        let image = UIImage(named: "image.png")!
        imageView.image = image
        let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        print("buffer[0]=\(pixelBuffer[0]), buffer[1]=\(pixelBuffer[1]), buffer[10]=\(pixelBuffer[10])")
        
        /// https://github.com/pytorch/pytorch/issues/66993
        let copiedBufferPtr = UnsafeMutablePointer<Float>.allocate(capacity: pixelBuffer.count)
        copiedBufferPtr.initialize(from: pixelBuffer, count: pixelBuffer.count)
        guard let outputs = module.predict(image: copiedBufferPtr) else {
            copiedBufferPtr.deallocate()
            return
        }
        copiedBufferPtr.deallocate()
        
        print("outputs[0]=\(outputs[0].doubleValue), outputs[1]=\(outputs[1].doubleValue), outputs[10]=\(outputs[10].doubleValue)")
        
        let zippedResults = zip(labels.indices, outputs)
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
        var text = ""
        for result in sortedResults {
            text += "\u{2022} \(labels[result.0]) \n\n"
        }
        resultView.text = text
        
//        moon_test()
    }
    
    private func moon_test() {
        
        print("=============moon test============")
        
        let image = UIImage(named: "image2.png")!
        let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        print("buffer[0]=\(pixelBuffer[0]), buffer[1]=\(pixelBuffer[1]), buffer[10]=\(pixelBuffer[10])")
        
        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
            return
        }
        print("outputs[0]=\(outputs[0].doubleValue), outputs[1]=\(outputs[1].doubleValue), outputs[10]=\(outputs[10].doubleValue)")
    }
}
