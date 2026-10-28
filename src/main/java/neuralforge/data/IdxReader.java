package neuralforge.data;

import neuralforge.math.Matrix;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

public class IdxReader{

    private static final int TYPE_UNSIGNED_BYTE = 0x08;
    private static final int MAX_DIMENSIONS = 4;

    private IdxReader(){
    }

    public static class IdxArray{

        private final int[] dims;
        private final byte[] data;

        IdxArray(int[] dims, byte[] data){
            this.dims = dims;
            this.data = data;
        }

        public int[] getDims(){
            return dims.clone();
        }

        public byte[] getData(){
            return data;
        }
    }

    public static IdxArray read(String path, int limit){
        try(DataInputStream in = open(path)){

            int zeroA = in.readUnsignedByte();
            int zeroB = in.readUnsignedByte();
            int type = in.readUnsignedByte();
            int dimensionCount = in.readUnsignedByte();

            if(zeroA != 0 || zeroB != 0){
                throw new IllegalArgumentException(path + " is not an IDX file (the first two bytes must be 0)");
            }
            if(type != TYPE_UNSIGNED_BYTE){
                throw new IllegalArgumentException(path + " has value type " + type + ", only unsigned bytes (8) are supported");
            }
            if(dimensionCount < 1 || dimensionCount > MAX_DIMENSIONS){
                throw new IllegalArgumentException(path + " claims to have " + dimensionCount + " dimensions");
            }

            int[] dims = new int[dimensionCount];
            for(int i = 0; i < dimensionCount; i++){
                dims[i] = in.readInt();
                if(dims[i] <= 0){
                    throw new IllegalArgumentException(path + " has a dimension of size " + dims[i]);
                }
            }

            int fileCount = dims[0];
            int count = fileCount;
            if(limit > 0 && limit < fileCount){
                count = limit;
            }

            long itemSize = 1;
            for(int i = 1; i < dimensionCount; i++){
                itemSize *= dims[i];
                if(itemSize > Integer.MAX_VALUE){
                    throw new IllegalArgumentException(path + " has items that are too large");
                }
            }

            long totalBytes = count * itemSize;
            if(totalBytes > Integer.MAX_VALUE){
                throw new IllegalArgumentException(path + " is too large to load at once, use a limit");
            }

            byte[] data = new byte[(int) totalBytes];
            in.readFully(data);

            if(count == fileCount && in.read() != -1){
                throw new IllegalArgumentException(path + " has more data than its header says");
            }

            dims[0] = count;
            return new IdxArray(dims, data);
        }catch(EOFException e){
            throw new IllegalArgumentException(path + " ended too early, the file is damaged or incomplete");
        }catch(IOException e){
            throw new UncheckedIOException("could not read " + path + ": " + e.getMessage(), e);
        }
    }

    public static Matrix readImages(String path, int limit){
        IdxArray array = read(path, limit);
        int[] dims = array.getDims();
        if(dims.length != 3){
            throw new IllegalArgumentException(path + " has " + dims.length + " dimensions, an image file needs 3 (count, rows, columns)");
        }

        int count = dims[0];
        int pixels = dims[1] * dims[2];
        byte[] data = array.getData();

        Matrix images = new Matrix(count, pixels);
        for(int i = 0; i < count; i++){
            for(int j = 0; j < pixels; j++){
                int value = data[i * pixels + j] & 0xFF;
                images.set(i, j, value / 255.0);
            }
        }
        return images;
    }

    public static int[] readLabels(String path, int limit){
        IdxArray array = read(path, limit);
        int[] dims = array.getDims();
        if(dims.length != 1){
            throw new IllegalArgumentException(path + " has " + dims.length + " dimensions, a label file needs 1");
        }

        byte[] data = array.getData();
        int[] labels = new int[dims[0]];
        for(int i = 0; i < labels.length; i++){
            labels[i] = data[i] & 0xFF;
        }
        return labels;
    }

    private static DataInputStream open(String path) throws IOException{
        InputStream raw = new BufferedInputStream(Files.newInputStream(Paths.get(path)));
        try{
            raw.mark(2);
            int first = raw.read();
            int second = raw.read();
            raw.reset();

            InputStream source = raw;
            if(first == 0x1f && second == 0x8b){
                source = new BufferedInputStream(new GZIPInputStream(raw));
            }
            return new DataInputStream(source);
        }catch(IOException e){
            raw.close();
            throw e;
        }
    }
}
